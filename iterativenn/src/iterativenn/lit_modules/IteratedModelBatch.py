import time
from typing import Dict, List

import omegaconf
import torch
from asteval import Interpreter
from pytorch_lightning import LightningModule
from torch.optim import lr_scheduler

from iterativenn.lit_modules.IteratedModel import IteratedModelCallbacks

aeval = Interpreter()


class IteratedModelNLP(LightningModule):
    def __init__(self, model, callbacks,
                 normalize_loss=False, optimizer='Adam'):
        """
        A wrapper for a model that handles the training loop for a sequence of inputs.
        In particular, the responsibilty of this class it to handle translating from
        a DataLoader (with all of its batched and shuffling) to a sequence of inputs.

        Args:
            model (pytorch.Module): A model that takes a minibatch of inputs and returns a minibatch of outputs for each input.
            loss_func (function): A function that takes a list of truths and a list of predictions and returns a loss.
        """
        super().__init__()
        self.model = model
        self.callbacks = callbacks

        # These are just convenience flags.  One could imagine having these in a
        # subclass, but I don't think that is necessary.
        self.normalize_loss = normalize_loss
        self.optimizer = optimizer

    def forward(self, forward_item_batch: torch.Tensor) -> torch.Tensor:
        """
            Args:
                forward_item_batch: A batch of input sequences, where each input sequence is a list of
                    tensors with shape (seq_len, input_dim).

            Returns:
                torch.Tensor: A tensor with shape (batch_size, seq_len, z_output_size), containing the model outputs for each
                    input sequence in the batch.
        """
        x_batch = forward_item_batch
        z_batch = self.callbacks.initialization(device=x_batch[0][0].device)
        for i in range(x_batch.shape[1]):
            z_i = self.callbacks.data(z_batch, x_batch, i)
            z_i = self.model(z_i[:, i, :])
            z_batch = torch.cat([z_batch[:, :i, :], z_i.unsqueeze(1), z_batch[:, i + 1:, :]], dim=1)

        return self.callbacks.output(z_batch)

    def training_step(self, train_item_batch, batch_idx, do_logging=True):
        x_batch, y_batch = train_item_batch
        x_batch.requires_grad_()
        y_batch.requires_grad_()
        z_hat_batch = self(x_batch)
        batch_sequence_loss = 0
        for i in range(x_batch.shape[1]):
            batch_sequence_loss += self.callbacks.loss(z_hat_batch, y_batch, i)
        if do_logging:
            self.log("training_loss", batch_sequence_loss,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(train_item_batch))
        return batch_sequence_loss

    def validation_step(self, validation_item_batch, batch_idx, do_logging=True):
        x_batch, y_batch = validation_item_batch
        z_hat_batch = self(x_batch)
        batch_sequence_loss = 0
        for i in range(x_batch.shape[1]):
            batch_sequence_loss += self.callbacks.loss(z_hat_batch, y_batch, i)

        # boolean mask tensor, to ignore indexes that were not masked.
        mask = (y_batch != -100.0) # set to true
        y_batch_masked = y_batch[mask]
        z_hat_batch_masked = z_hat_batch[mask]
        acc = torch.mean((z_hat_batch_masked == y_batch_masked).float())


        if do_logging:
            self.log("validation_loss", batch_sequence_loss,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(validation_item_batch))
            self.log("validation_acc", acc,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(validation_item_batch))
        return batch_sequence_loss

    def test_step(self, test_item_batch, batch_idx, do_logging=True):
        x_batch, y_batch = test_item_batch
        z_hat_batch = self(x_batch)
        batch_sequence_loss = 0
        for i in range(x_batch.shape[1]):
            batch_sequence_loss += self.callbacks.loss(z_hat_batch, y_batch, i)
        if do_logging:
                self.log("test_loss", batch_sequence_loss,
                         on_step=False,
                         on_epoch=True,
                         batch_size=len(test_item_batch))
        return batch_sequence_loss

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()
        return super().on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        self.epoch_end_time = time.time()
        self.log("epoch_time", self.epoch_end_time - self.epoch_start_time, on_step=False, on_epoch=True)
        return super().on_train_epoch_start()

    def _init_param_groups(self) -> List[Dict]:
        """Initialize the parameter groups. Used to ensure lr is applied to a specified parameter group by the user
         when we initialize the optimizer.
        Help:
            torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0,
            weight_decay=0, nesterov=False, *, maximize=False, foreach=None, differentiable=False)
        Returns:
            List[Dict]: A list of parameter group dictionaries.
        """
        params = []
        for i in self.model.blocks:
            params.append({"params": self.model.blocks[i].parameters(), 'lr': self.callbacks.optimization()[i]})
        return params

    def configure_optimizers(self):
        # IMPORTANT: You need to be careful with momentum methods in this kind of
        # experiment.  The momentum is a function of the training history, and will
        # get reset when you make a new model.  This will change the training
        # behavior.
        if self.optimizer == 'Adam':
            return torch.optim.Adam(params=self.parameters(), lr=0.02)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(params=self.parameters(), lr=0.02)
        elif self.optimizer == 'Adam_customLR':
            optimizer = torch.optim.Adam(params=self._init_param_groups())
            # this scheduler will work per param group
            scheduler = {
                "scheduler": lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1),
                "interval": "epoch",
            }
            return [optimizer], [scheduler]
        elif self.optimizer == 'SGD_customLR':
            optimizer = torch.optim.SGD(params=self._init_param_groups())
            # this scheduler will work per param group
            scheduler = {
                "scheduler": lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1),
                "interval": "epoch",
            }
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def number_of_trainable_parameters(self):
        if hasattr(self.model, "number_of_trainable_parameters"):
            return self.model.number_of_trainable_parameters()
        else:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConfigCallbacks(IteratedModelCallbacks):
    def __init__(self, cfg):
        # This bears some explanation.
        # The config file is a yaml file that is read in by hydra.cc.
        # This is a very powerful, but only allows for a limited set of data types.
        # On the other hand, a dict is a very flexible data type and
        # contain arbitrary python objects.
        # So, I convert the config file to a dict and then use asteval to
        # create the objects that I need.
        # I extend the data types by allowing strings that begin with "asteval::"
        # to be evaluated by asteval.  This allows me to do things like
        # asteval::torch.randn(10, 10)
        if isinstance(cfg, dict):
            dict_cfg = cfg
        elif isinstance(cfg, omegaconf.dictconfig.DictConfig):
            dict_cfg = omegaconf.OmegaConf.to_container(cfg)
        else:
            raise ValueError(f"Unknown config type: {type(cfg)}")

        # So, this processes the config values to see if there are any
        # strings that need to be run through asteval.
        def dict_asteval(dict_cfg):
            for key, value in dict_cfg.items():
                if isinstance(value, dict):
                    dict_asteval(value)
                elif isinstance(value, str):
                    if value.startswith("asteval::"):
                        command = value.replace("asteval::", "")
                        new_value = aeval(command)
                        dict_cfg[key] = new_value

        # At this point, we call the recursive function to process the config.
        dict_asteval(dict_cfg)
        # and all of the objects are now in the dict.
        # dict_cfg

        # and now we can use the config as normal with the strings replaced.
        self.loss_ = ConfigCallbacks.loss_func_factory(dict_cfg["loss"])
        self.initialization_ = ConfigCallbacks.initialization_func_factory(dict_cfg["initialization"])
        self.data_ = ConfigCallbacks.data_func_factory(dict_cfg["data"])
        self.output_ = ConfigCallbacks.output_func_factory(dict_cfg["output"])
        try:
            # TODO: this is optional, we can make this as required in configs
            self.optimization_ = ConfigCallbacks.optimization_func_factory(dict_cfg["optimization"])
        except KeyError:
            print(f"No optimization routine provided")

    def initialization(self, device):
        return self.initialization_(device)

    def loss(self, z, y, sequence_idx):
        return self.loss_(z, y, sequence_idx)

    def data(self, z, x, sequence_idx):
        return self.data_(z, x, sequence_idx)

    def output(self, z):
        return self.output_(z)

    def optimization(self):
        return self.optimization_()

    @staticmethod
    def loss_func_factory(cfg):
        if cfg["func"] == "CrossEntropyLoss":
            def loss_func(z, y, sequence_idx):
                assert cfg['sequence_position'] in ['last', 'first',
                                                    'all'], "sequence_position must be one of ['last', 'first', 'all']"

                if cfg['sequence_position'] == 'first' and sequence_idx != 0:
                    return 0
                else:
                    loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(z.float(), y.float())
                    return loss
            return loss_func
        elif cfg["func"] == "MSELoss":
            def loss_func(z, y, sequence_idx):
                # for len(y_sequence) one can use y shape
                assert cfg['sequence_position'] in ['last', 'first',
                                                    'all'], "sequence_position must be one of ['last', 'first', 'all']"
                if cfg['sequence_position'] == 'first' and sequence_idx != 0:
                    return 0
                else:
                    return torch.nn.MSELoss()(z.float(), y.float())
            return loss_func
        else:
            raise ValueError(f"Unknown loss function {cfg['func']}")

    @staticmethod
    def initialization_func_factory(cfg):
        if cfg["func"] == "zeros":
            def initialization_func(device):
                return torch.zeros((cfg["batch_size"], cfg["seq_len"], cfg["size"]), device=device)

            return initialization_func
        else:
            raise ValueError(f"Unknown initialization function {cfg['func']}")

    @staticmethod
    def data_func_factory(cfg):
        if cfg["func"] == "insert":
            def data_func(z, x, sequence_idx, flatten_input=cfg['flatten_input']):
                with torch.no_grad():
                    if flatten_input:
                        raise NotImplementedError
                    else:
                        z[:, sequence_idx, cfg["idx_list"]] = x[:, sequence_idx, :]
                    return z

            return data_func
        else:
            raise ValueError(f"Unknown data function {cfg['func']}")

    @staticmethod
    def output_func_factory(cfg):
        if cfg["func"] == "max":
            def output_func(z):
                return torch.argmax(z[:, :, cfg['idx_list']], dim=2)

            return output_func
        elif cfg["func"] == "all":
            def output_func(z):
                return z[:, :, cfg['idx_list']]

            return output_func
        else:
            raise ValueError(f"Unknown output function {cfg['func']}")

    @staticmethod
    def optimization_func_factory(cfg):
        if cfg["func"] == "customLR":
            def optimization_func():
                blocks_lr_dict = {}
                for i, row in enumerate(cfg["block_lr"]):
                    for j, lr_v in enumerate(row):
                        blocks_lr_dict.update({str((i, j)): lr_v})
                return blocks_lr_dict

            return optimization_func
        else:
            raise ValueError(f"Unknown optimization function {cfg['func']}")

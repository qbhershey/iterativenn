import time
from typing import Any, Dict, List

import omegaconf
import torch
from asteval import Interpreter
from pytorch_lightning import LightningModule
from torch.optim import lr_scheduler

from iterativenn.lit_modules.IteratedModel import IteratedModelCallbacks

aeval = Interpreter()

class IteratedModelGPT2(LightningModule):
    def __init__(self, model, callbacks, input_layer,
                 normalize_loss=False, optimizer='Adam', iterations=None):
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
        self.input_layer = input_layer
        # These are just convenience flags.  One could imagine having these in a
        # subclass, but I don't think that is necessary.
        self.normalize_loss = normalize_loss
        self.optimizer = optimizer
        self.validation_step_outputs = []
        self.iterations = iterations

    def forward(self, item_batch: torch.Tensor) -> torch.Tensor:
        """
            Args:
                forward_item_batch: A batch of input sequences, where each input sequence is a list of
                    tensors with shape (seq_len, input_dim).

            Returns:
                torch.Tensor: A tensor with shape (batch_size, seq_len, z_output_size), containing the model outputs for each
                    input sequence in the batch.

        """
        # <B, seq_len, hidden_size>
        x = self.input_layer(**item_batch)
        z = self.callbacks.initialization(device=x[0][0].device)
        for i in range(self.iterations):
            z = self.callbacks.data(z, x, i)
            z = self.model(z)
        z_out = self.model(z)
        logits = self.callbacks.output(z_out)
        return logits

    def training_step(self, item_batch, batch_idx, do_logging=True):
        labels = item_batch['labels']
        y_hat_batch = self(item_batch)
        i = 1 # TODO
        batch_sequence_loss , ppx, acc= self.callbacks.loss(y_hat_batch, labels, i)
        if do_logging:
            self.log("training_loss", batch_sequence_loss,
                     on_step=True,
                     on_epoch=True,
                     batch_size=len(item_batch),
                     sync_dist=True)
            self.log("training_ppx", ppx,
                     on_step=True,
                     on_epoch=True,
                     batch_size=len(item_batch),
                     sync_dist=True)
            self.log("training_acc", acc,
                     on_step=True,
                     on_epoch=True,
                     batch_size=len(item_batch),
                     sync_dist=True)
        return batch_sequence_loss

    def validation_step(self, item_batch, batch_idx, do_logging=True):
        labels = item_batch['labels']
        y_hat_batch = self(item_batch)
        i = 1 # TODO
        batch_sequence_loss, ppx, acc, = self.callbacks.loss(y_hat_batch, labels, i)
        if do_logging:
            self.log("validation_loss", batch_sequence_loss,
                     on_step=True,
                     on_epoch=True,
                     batch_size=len(item_batch),
                     sync_dist=True)
            self.log("validation_ppx", ppx,
                     on_step=True,
                     on_epoch=True,
                     batch_size=len(item_batch),
                     sync_dist=True)
            self.log("validation_acc", acc,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(item_batch),
                     sync_dist=True)
        self.validation_step_outputs.append(batch_sequence_loss)
        return batch_sequence_loss

    def on_validation_epoch_end(self):
        # outs is a list of whatever you returned in `validation_step`
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss_agg", loss)
        self.validation_step_outputs.clear()

    def test_step(self, item_batch, batch_idx, do_logging=True):
        labels = item_batch['lables']
        y_hat_batch = self(item_batch)
        i = 1 # TODO
        batch_sequence_loss = self.callbacks.loss(y_hat_batch, labels, i)
        if do_logging:
            self.log("training_loss", batch_sequence_loss,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(item_batch))
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
            optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-5)
            #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

            return optimizer
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

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(batch)



class IteratedModelGPT2CachedGrow(LightningModule):
    def __init__(self, model, callbacks, input_layer,
                 normalize_loss=False, optimizer='Adam', iterations=None):
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
        self.input_layer = input_layer
        # These are just convenience flags.  One could imagine having these in a
        # subclass, but I don't think that is necessary.
        self.normalize_loss = normalize_loss
        self.optimizer = optimizer
        self.validation_step_outputs = []
        self.train_cache = []
        self.validation_cahe = []
        self.iterations = iterations

    def forward(self, item_batch: torch.Tensor) -> torch.Tensor:
        """
            Args:
                forward_item_batch: A batch of input sequences, where each input sequence is a list of
                    tensors with shape (seq_len, input_dim).
            Returns:
                torch.Tensor: A tensor with shape (batch_size, seq_len, z_output_size), containing the model outputs for each
                    input sequence in the batch.
        """
        z = self.callbacks.initialization(device=item_batch[0][0].device)
        for i in range(self.iterations):
            z = self.callbacks.data(z, item_batch, i)
        z_out = self.model(z)
        logits = self.callbacks.output(z_out)
        return logits

    def training_step(self, item_batch, batch_idx, do_logging=True):
        labels = item_batch['labels']
        print(self.train_cache['ids'])
        item_batch_cached = self.train_cache['values'][self.train_cache['ids'].index(batch_idx)]
        y_hat_batch = self(item_batch_cached)
        i = 1 # TODO
        batch_sequence_loss , ppx, acc= self.callbacks.loss(y_hat_batch, labels, i)
        if do_logging:
            self.log("training_loss", batch_sequence_loss,
                     on_step=True,
                     on_epoch=True,
                     batch_size=len(item_batch),
                     sync_dist=True)
            self.log("training_ppx", ppx,
                     on_step=True,
                     on_epoch=True,
                     batch_size=len(item_batch),
                     sync_dist=True)
            self.log("training_acc", acc,
                     on_step=True,
                     on_epoch=True,
                     batch_size=len(item_batch),
                     sync_dist=True)
        return batch_sequence_loss

    def validation_step(self, item_batch, batch_idx, do_logging=True):
        labels = item_batch['labels']
        item_batch_cached = self.train_cache['values'][self.train_cache['ids'].index(batch_idx)]
        y_hat_batch = self(item_batch_cached)
        i = 1 # TODO
        batch_sequence_loss, ppx, acc, = self.callbacks.loss(y_hat_batch, labels, i)
        if do_logging:
            self.log("validation_loss", batch_sequence_loss,
                     on_step=True,
                     on_epoch=True,
                     batch_size=len(item_batch),
                     sync_dist=True)
            self.log("validation_ppx", ppx,
                     on_step=True,
                     on_epoch=True,
                     batch_size=len(item_batch),
                     sync_dist=True)
            self.log("validation_acc", acc,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(item_batch),
                     sync_dist=True)
        self.validation_step_outputs.append(batch_sequence_loss)
        return batch_sequence_loss

    def on_validation_epoch_end(self):
        # outs is a list of whatever you returned in `validation_step`
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss_agg", loss)
        self.validation_step_outputs.clear()

    def test_step(self, item_batch, batch_idx, do_logging=True):
        # TODO update logic with cache
        labels = item_batch['lables']
        item_batch_cached = self.train_cache['values'][self.train_cache['ids'].index(batch_idx)]
        y_hat_batch = self(item_batch_cached)
        i = 1 # TODO
        batch_sequence_loss = self.callbacks.loss(y_hat_batch, labels, i)
        if do_logging:
            self.log("training_loss", batch_sequence_loss,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(item_batch))
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
            optimizer = torch.optim.Adam(params=self.parameters(), lr=1e-5)
            #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

            return optimizer
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
            def loss_func(logits, y, sequence_idx):
                assert cfg['sequence_position'] in ['last', 'first',
                                                    'all'], "sequence_position must be one of ['last', 'first', 'all']"

                if cfg['sequence_position'] == 'first' and sequence_idx != 0:
                    return 0
                else:

                    logits = logits.view(*cfg['logits_shape'])
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = y[..., 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    preds = torch.argmax(shift_logits, dim=-1)
                    acc =  torch.mean((preds == shift_labels).float())
                    loss_p = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
                    ppx = torch.exp(torch.mean(loss_p))
                    return loss, ppx, acc,
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
                return torch.zeros((cfg["batch_size"], cfg["size"]), device=device)

            return initialization_func
        else:
            raise ValueError(f"Unknown initialization function {cfg['func']}")

    @staticmethod
    def data_func_factory(cfg):
        if cfg["func"] == "insert":
            def data_func(z, x, sequence_idx, flatten_input=cfg['flatten_input']):
                with torch.no_grad():
                    if flatten_input:
                        z[:, cfg["idx_list"]] = x.view(cfg["batch_size"], len(cfg["idx_list"]))
                    else:
                        z[:, cfg["idx_list"]] = x
                    return z

            return data_func
        else:
            raise ValueError(f"Unknown data function {cfg['func']}")

    @staticmethod
    def output_func_factory(cfg):
        if cfg["func"] == "max":
            def output_func(z):
                return torch.argmax(z[:, cfg['idx_list']], dim=1)

            return output_func
        elif cfg["func"] == "all":
            def output_func(z):
                return z[:, cfg['idx_list']]

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

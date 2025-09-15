import time
from abc import ABC, abstractmethod
from typing import Dict, List

import omegaconf
import torch
from torch.nn.utils.rnn import pad_sequence
from asteval import Interpreter
from pytorch_lightning import LightningModule
from torch.optim import lr_scheduler
import logging
import time
logger = logging.getLogger(__name__)

aeval = Interpreter()

class IteratedModel(LightningModule):
    def __init__(self, model, callbacks, normalize_loss=True, optimizer='Adam', learning_rate=0.02, batch_optimize=True):
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
        self.learning_rate = learning_rate
        self.batch_optimize = batch_optimize
        self.validation_step_outputs = []
        self.train_step_outputs = []
        # Keep a copy of the losses
        self.losses = []

    def forward(self, forward_item_batch):
        #Discontinued this for faster batch optimized loop
        output = []
        for forward_item in forward_item_batch:
            x_sequence, y_sequence = forward_item["x"], forward_item["y"]
            z_sequence = []
            z = self.callbacks.initialization(x_sequence[0].device)
            for sequence_idx, (x, y) in enumerate(zip(x_sequence, y_sequence)):
                z = self.callbacks.data(z, x)
                z = self.model(z)
                z_sequence.append(z)
            # This is the output function that is used to convert the output of the model to a 
            # human readable form.  Note, we cannot assume any particular form for the output of the model.
            # so we just pack it into a list and return it.
            output += [self.callbacks.output(z, sequence_idx, x_sequence, y_sequence, z_sequence)]
        return output

    def forward(self, item_batch):
        output = self.process_sequence_without_loss_batch_optimize_(item_batch)
        return output
        
    def process_sequence_without_loss_batch_optimize_(self, item_batch):
        output, x_sequence, y_sequence, z_sequence = [], [], [], []
        # We assume that item_batch is a list of dictionaries with the keys "x" and "y".
        # The length of the list is the length of the sequence.
        for item in item_batch:
            assert len(item['x']) == len(item['y']), "The length of the input and output sequences must be the same."
            # The values of "x" and "y" are arbitrary objects that can be converted to tensors.  
            for i in range(len(item['x'])):
                item['x'][i] = torch.as_tensor(item['x'][i])
                item['y'][i] = torch.as_tensor(item['y'][i])    

        # Get the device that the data is on.
        device = item_batch[0]['x'][0].device

        # Check the input data.  We want to support a variety of input shapes and types.
        if item_batch[0]['x'][0].dim() == 0:
            x_slice = torch.zeros((len(item_batch), 1), device=device)
        else:
            x_slice = torch.zeros([len(item_batch)] + list(item_batch[0]['x'][0].shape), device=device)

        if item_batch[0]['y'][0].dim() == 0:
           y_slice = torch.zeros((len(item_batch), 1), device=device) * torch.nan
        else:
           y_slice = torch.zeros([len(item_batch)] + list(item_batch[0]['y'][0].shape), device=device) * torch.nan

        # Figure out the lengths of all the sequences.
        sequence_lengths = [len(item['x']) for item in item_batch]        

        # Initialize the input vector
        z = self.callbacks.initialization(item_batch[0]['x'][0].device, rows=len(item_batch))

        # Iterate over the sequences.  This can be thought of as a loop over time.
        for sequence_idx in range(max(sequence_lengths)):
            # Gather all of the sequences into a single batch.
            # Note, not all sequences are the same length, so we need to pad them.

            #####################################################
            # Rest the slices to 0 and nan, which are the default values
            # for padding out the sequence past were it is defined.
            x_slice *= 0
            y_slice *= torch.nan
            # Now, fill in the values for the current time step
            for item_idx, item in enumerate(item_batch):
                # We only fill in the values if the input sequence is long enough.
                if sequence_idx < len(item['x']):
                    x_slice[item_idx, :] = item['x'][sequence_idx]
                    y_slice[item_idx, :] = item['y'][sequence_idx]
            #####################################################
            
            # Now, these functions are the core of the model. They
            # use the packed data to compute the loss and update the model.
            z = self.callbacks.data(z, x_slice)
            z = self.model(z)
            x_sequence += [x_slice]
            y_sequence += [y_slice]
            z_sequence += [z]

        outputs = self.callbacks.output(z, sequence_idx, x_sequence, y_sequence, z_sequence)

        for r in range(len(outputs)):
            output += [torch.unsqueeze(outputs[r], 0)]
            
        return output

    def process_sequence_with_loss_(self, item_batch, batch_idx, do_logging):        
        if self.batch_optimize:
            batch_loss = self.process_sequence_with_loss_batch_optimize_(item_batch, batch_idx, do_logging)
        else:
            batch_loss = self.process_sequence_with_loss_batch_iterate_(item_batch, batch_idx)
        return batch_loss

    def process_sequence_with_loss_batch_optimize_(self, item_batch, batch_idx, do_logging):
        # We assume that item_batch is a list of dictionaries with the keys "x" and "y".
        # The length of the list is the length of the sequence.
        for item in item_batch:
            assert len(item['x']) == len(item['y']), "The length of the input and output sequences must be the same."
            # The values of "x" and "y" are arbitrary objects that can be converted to tensors.  
            for i in range(len(item['x'])):
                item['x'][i] = torch.as_tensor(item['x'][i])
                item['y'][i] = torch.as_tensor(item['y'][i])    

        # Get the device that the data is on.
        device = item_batch[0]['x'][0].device

        # Check the input data.  We want to support a variety of input shapes and types.
        if item_batch[0]['x'][0].dim() == 0:
            x_slice = torch.zeros((len(item_batch), 1), device=device)
        else:
            x_slice = torch.zeros([len(item_batch)] + list(item_batch[0]['x'][0].shape), device=device)

        if item_batch[0]['y'][0].dim() == 0:
           y_slice = torch.zeros((len(item_batch), 1), device=device) * torch.nan
        else:
           y_slice = torch.zeros([len(item_batch)] + list(item_batch[0]['y'][0].shape), device=device) * torch.nan

        # Figure out the lengths of all the sequences.
        sequence_lengths = [len(item['x']) for item in item_batch]        
        batch_loss = 0.0
        batch_num_valid_loss = 0

        # Initialize the input vector
        z = self.callbacks.initialization(item_batch[0]['x'][0].device, rows=len(item_batch))

        # Iterate over the sequences.  This can be thought of as a loop over time.
        for sequence_idx in range(max(sequence_lengths)):
            # Gather all of the sequences into a single batch.
            # Note, not all sequences are the same length, so we need to pad them.

            #####################################################
            # Rest the slices to 0 and nan, which are the default values
            # for padding out the sequence past were it is defined.
            x_slice *= 0
            y_slice *= torch.nan
            # Now, fill in the values for the current time step
            for item_idx, item in enumerate(item_batch):
                # We only fill in the values if the input sequence is long enough.
                if sequence_idx < len(item['x']):
                    x_slice[item_idx, :] = item['x'][sequence_idx]
                    y_slice[item_idx, :] = item['y'][sequence_idx]
            #####################################################
            
            # Now, these functions are the core of the model. They
            # use the packed data to compute the loss and update the model.
            z = self.callbacks.data(z, x_slice)

            z = self.model(z)
            if self.logger is not None and hasattr(self.logger, 'log_z'):
                self.logger.log_z('z', z, sequence_idx, batch_idx, self.global_step)

            current_loss = self.callbacks.loss(z, y_slice)
            correct, num = self.callbacks.accuracy(z, y_slice)
            self.correct += correct
            self.num += num            
            
            #if do_logging: self.log(f"sequence_idx_{sequence_idx}_loss", current_loss, on_step=True, on_epoch=False)
            batch_loss += current_loss

            # This counts the number of places where the loss is valid, so we 
            # can normalize the loss later if we want.
            batch_num_valid_loss += torch.count_nonzero(~torch.isnan(y_slice))

        if torch.isnan(batch_loss).any():
            raise ValueError("Nan loss not allowed")

        if self.normalize_loss:
            batch_loss /= batch_num_valid_loss
        return batch_loss
      
    def process_sequence_with_loss_batch_iterate_(self, item_batch, batch_idx):   
        raise NotImplementedError("process_sequence_with_loss_batch_iterate_ is not maintained.  Use process_sequence_with_loss_batch_optimize_ instead.")
        batch_loss = 0.0
        batch_num_valid_loss = 0
        batch_size = 0
        for item_idx, item in enumerate(item_batch):
            sequence_loss = 0
            # Note, if the item does not have the correct keys this will fail.
            x_sequence, y_sequence = item["x"], item["y"]
            assert (len(x_sequence)) > 0 and (len(y_sequence) > 0), "The length of the x and y sequences must be greater than 0"
            assert len(x_sequence) == len(y_sequence), "The length of the x and y sequences must be the same"
            # This is the initialization of the sequence needs the device since pytorch lightning does not
            # reach this far into the code.
            z = self.callbacks.initialization(x_sequence[0].device)
            for sequence_idx, (x, y) in enumerate(zip(x_sequence, y_sequence)):
                ####################################################
                # NOTE:  The idea here is that 
                #   callbacks.data depends on external data (e.g., the input sequence)
                #   model depends on the previous output and nothing else.  Internally model can be a sequence of models.
                #   callbacks.loss depends on external data (e.g., the output sequence)
                # Is this the right way to do this?  For example, there could be a single model that takes the entire sequence
                # and handles all of this internally.  I worry this would be too much of a black box, and not the
                # pytorch lightning way of doing things. This works for the moment, and I don't have a use-case
                # for the other way at the moment.  Once I have a use-case I will revisit this code.
                ####################################################
                z = self.callbacks.data(z, x)
                z = self.model(z)
                current_loss = self.callbacks.loss(z, y)
                self.log(f"sequence_idx_{sequence_idx}_loss", current_loss, on_step=True, on_epoch=False)
                sequence_loss += current_loss
                batch_num_valid_loss += torch.count_nonzero(~torch.isnan(torch.tensor(y)))
                batch_size += 1
            batch_loss += sequence_loss
        if torch.isnan(batch_loss).any():
            raise ValueError("Nan loss not allowed")
        if self.normalize_loss:
            batch_loss /= batch_num_valid_loss
        return batch_loss

    def training_step(self, train_item_batch, batch_idx, do_logging=True):
        batch_loss = self.process_sequence_with_loss_(train_item_batch, batch_idx, do_logging=do_logging)
        if do_logging:
            self.log("training_loss", batch_loss, 
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(train_item_batch))
        self.losses.append(batch_loss)
        self.train_step_outputs.append(batch_loss.cpu())
        return batch_loss

    def validation_step(self, validation_item_batch, batch_idx, do_logging=True):
        batch_loss = self.process_sequence_with_loss_(validation_item_batch, batch_idx, do_logging=do_logging)
        if do_logging:
            self.log("validation_loss", batch_loss,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(validation_item_batch))
        self.validation_step_outputs.append(batch_loss)
        return batch_loss

    def test_step(self, test_item_batch, batch_idx, do_logging=True):
        batch_loss = self.process_sequence_with_loss_(test_item_batch, batch_idx, do_logging=do_logging)
        if do_logging:
            self.log("test_loss", batch_loss,
                     on_step=False,
                     on_epoch=True,
                     batch_size=len(test_item_batch))
        return batch_loss

    # This happens at the start of the epoch
    def on_train_epoch_start(self) -> None:
        self.training_epoch_start_time = time.time()
        self.correct, self.num = 0, 0

    # This happens at the end of the epoch, just before the next function
    def on_train_epoch_end(self) -> None:
        self.epoch_end_time = time.time()
        if self.correct == 0 and self.num == 0: self.num = 1
        if self.logger is not None:
            self.logger.log_metrics({
                "IterateModel_training_epoch_time": self.epoch_end_time - self.training_epoch_start_time,
                "IterateModel_training_epoch_average_loss": torch.stack(self.train_step_outputs).mean(),
                "IterateModel_training_epoch": self.current_epoch,
                "IterateModel_training_epoch_accuracy": self.correct / self.num
            }, self.current_epoch)
        self.train_step_outputs.clear()

    # This happens at the start of the epoch
    def on_validation_epoch_start(self) -> None:
        self.validation_epoch_start_time = time.time()
        self.correct, self.num = 0, 0

    # This happens at the end of the epoch, just before the next function
    def on_validation_epoch_end(self) -> None:
        self.epoch_end_time = time.time()
        if self.correct == 0 and self.num == 0: self.num = 1
        if self.logger is not None:
            self.logger.log_metrics({
                "IterateModel_valdiation_epoch_time": self.epoch_end_time - self.validation_epoch_start_time,
                "IterateModel_validation_epoch_average_loss": torch.stack(self.validation_step_outputs).mean(),
                "IterateModel_validation_epoch": self.current_epoch,
                "IterateModel_validation_epoch_accuracy": self.correct / self.num
            }, self.current_epoch)
        self.validation_step_outputs.clear()
    
    def on_test_epoch_start(self) -> None:
        self.correct, self.num = 0, 0

    def on_test_epoch_end(self) -> None:
        if self.correct == 0 and self.num == 0: self.num = 1
        if self.logger is not None:
            self.logger.log_metrics({
                "IterateModel_test_epoch_accuracy": self.correct / self.num
            }, self.current_epoch)        

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
            return torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(params=self.parameters(), lr=self.learning_rate)
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
    
class IteratedModelCallbacks(ABC):
    """Just to make things clean I wrap the callbacks in a class.  I suppose the abstract class is 
    not necessary, but it also makes it clear what the interface is.  It also allows for some
    documentation of the interface.

    In some sense, this interface is quite simple.
    initialization:  This function returns the initial value of $z$ for the model.  
    loss: This depends on $z$ and $y$ and returns a scalar loss function.
    data: This depends on $z$ and $x$ and returns a new $z$.
    output: This depends on $z$ and returns the output of the model.
    """
    @abstractmethod
    def initialization(self, device, rows):
        """This function returns the initial value of $z$ for the model.  This is called once at the 
        beginning of each sequence.  Note, this class might maintain internal state, so this function
        might not always return the same value.  For example, it can return the value from a previous call.
        Args:
            device (device):  This is a pytorch device.  This function will likely need to allocate data
                on this device.  NOTE:  pytorch lightning normally handles this for you, but I am not sure
                how to do this with the sequence model.
            rows (int):  How many rows to have in the initialized z.
        """
        pass

    @abstractmethod
    def accuracy(self, z, y):
        """
        This function returns a *scalar* accuracy reporting function.  This will be used by pytorch lightning to
        update the model.
        Args:
            z (Tensor): the output of the model.  This is the full $z$ from the model.
            y (Tensor): the target values.  This is the full $y$ from the target and this function can,
                in principle, be a very complicated (but differentiable) function of $y$ and $z$.
        """
        pass

    @abstractmethod
    def loss(self, z, y):
        """
        This function return a *scalar* loss function.  This will be used by pytorch lightning to
        update the model.
        Args:
            z (Tensor): the output of the model.  This is the full $z$ from the model.
            y (Tensor): the target values.  This is the full $y$ from the target and this function can,
                in principle, be a very complicated (but differentiable) function of $y$ and $z$.
        """
        pass

    @abstractmethod
    def data(self, z, x):
        """
        This function inserts the external data $x$ into the vector $z$.  This is where you would
        provide the external data to the model.  This function is called once for each element in the
        sequence.        
        Args:
            z (Tensor): the output of the model.  This is the full $z$ from the model.
            x (Tensor): the input data.  This is the $x$ provided by the data generator.  Note, the actual 
                $z$ that is returned can be an complex function of $x$ and $z$.  In fact, the function
                does not even need to be differentiable!    
        """
        pass

    @abstractmethod
    def output(self, z, sequence_idx, x_sequence, y_sequence, z_sequence):
        """This function returns some human readable output from the model. NOTE: Having just one
        function for this is not ideal, since it is not clear how to handle all possible outputs.  However,
        until a good use case arises, I am going to keep it simple.

        Args:
            z (Tensor): the output of the model.  This is the full $z$ from the model.
            sequence_idx (int): the index of the current element in the sequence.  This is useful if
                the data function depends on the sequence index.
            x_sequence (list):  the entire sequence of $x$ values.  This is useful if the output function
                depends on the entire sequence.
            y_sequence (list):  the entire sequence of $y$ values.  This is useful if the output function
                depends on the entire sequence.
            z_sequence (list):  the entire sequence of $z$ values.  This is useful if the output function
                depends on the entire sequence.
        """
        pass

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
        dict_cfg

        # and now we can use the config as normal with the strings replaced.
        self.accuracy_ = ConfigCallbacks.accuracy_func_factory(dict_cfg["loss"])   
        self.loss_ = ConfigCallbacks.loss_func_factory(dict_cfg["loss"])
        self.initialization_ = ConfigCallbacks.initialization_func_factory(dict_cfg["initialization"])
        self.data_ = ConfigCallbacks.data_func_factory(dict_cfg["data"])
        self.output_ = ConfigCallbacks.output_func_factory(dict_cfg["output"])
        try:
            # TODO: this is optional, we can make this as required in configs
            self.optimization_ = ConfigCallbacks.optimization_func_factory(dict_cfg["optimization"])
        except KeyError:
            logger.info("No optimization routine provided")            # print(f"No optimization routine provided")

    def initialization(self, device, rows=1):
        return self.initialization_(device, rows) 

    def accuracy(self, z, y):
        return self.accuracy_(z, y)

    def loss(self, z, y):
        return self.loss_(z, y)
    
    def data(self, z, x):
        return self.data_(z, x)

    def output(self, z, sequence_idx, x_sequence, y_sequence, z_sequence):
        return self.output_(z, sequence_idx, x_sequence, y_sequence, z_sequence)

    def optimization(self):
        return self.optimization_()

    @staticmethod
    def accuracy_func_factory(cfg):
        if cfg["func"] == "CrossEntropyLoss":
            def acc_func(z, y):
                with torch.no_grad():
                    y_computed = z[:, cfg['idx_list']]
                    if isinstance(y, torch.Tensor):
                        y_true = y.float()
                        if y_true.dim() == 1:
                            y_true = y_true.unsqueeze(1)
                    else:    
                        y_true = torch.tensor([y], device=z.device, dtype=torch.float).unsqueeze(1)
                    valid_idx = ~torch.isnan(y_true)
                    correct = torch.sum(torch.argmax(y_computed[valid_idx[:, 0], :], dim=1) == y_true[valid_idx[:,0], 0].long()).item()
                    num = valid_idx.sum().item()
                return correct, num
            return acc_func
            
        elif cfg["func"] == "MSELoss":
            def acc_func(z, y):
                return 0, 0
            return acc_func
            
        else:
            raise ValueError(f"Unknown accuracy function {cfg['func']}")

    @staticmethod
    def loss_func_factory(cfg):
        if cfg["func"] == "CrossEntropyLoss":
            def loss_func(z, y):
                # idx_list is another override, we only look at losses for the indices in idx_list
                # len(idx_list) must be equal to the number of catagories.
                y_computed = z[:, cfg['idx_list']]
                # We want to support a variety of input types for y
                if isinstance(y, torch.Tensor):
                    y_true = y.float()
                    if y_true.dim() == 1:
                        # We want to support y being a 1D tensor.  We add a dimension to make it 2D.
                        y_true = y_true.unsqueeze(1)
                else:    
                    # We want to support y being an int.  Note, we cast it to float since we need
                    # that to multiply by the nan values later.
                    y_true = torch.tensor([y], device=z.device, dtype=torch.float).unsqueeze(1)
                assert y_true.dim() == 2, "y_true must be a 2D tensor after processing"
                assert y_computed.dim() == 2, "y_computed must be a 2D tensor after processing"
                assert y_true.shape[0] == y_computed.shape[0], "y_true and y_computed must have the same number of rows after processing"
                assert y_true.shape[1] == 1, "y_true must have a single column after processing since it contains a true category index"

                # torch.nan is used as a flag to indicate that the loss is not to be computed
                valid_idx = ~torch.isnan(y_true)
                # If there are no valid indices, then we return 0.0
                if valid_idx.sum() == 0:
                    return 0.0
                # So, we only evaluate the loss on the indices where y_true is not nan
                # Also, we don't want to use the mean, since that will be affected by the number
                # of valid indices.  Instead, we use the sum, which will be the same regardless
                # of the number of valid indices.
                loss = torch.nn.CrossEntropyLoss(reduction='sum')(y_computed[valid_idx[:, 0], :], y_true[valid_idx[:,0], 0].long())
                if torch.isnan(loss):
                    # At this point if the loss is still nan, then we raise an error, since 
                    # that means there is some case that our code is not handling.
                    raise ValueError("Loss is nan")
                return loss 
            return loss_func
        elif cfg["func"] == "MSELoss":
            def loss_func(z, y):
                y_computed = z[:, cfg['idx_list']]
                # We want to support a variety of input types for y
                if isinstance(y, torch.Tensor):
                    y_true = y
                else:    
                    y_true = torch.tensor([y], device=z.device, dtype=torch.float)
                # we want to support many different use cases, so we allow y_true to be a vector
                # while y_computed is a matrix.  In this case, we need to reshape y_true to match
                y_true = y_true.reshape(y_computed.shape)

                assert y_true.dim() == 2, "y_true must be a 2D tensor after processing"
                assert y_computed.dim() == 2, "y_computed must be a 2D tensor after processing"

                # torch.nan is used as a flag to indicate that the loss is not to be computed
                valid_idx = ~torch.isnan(y_true)
                # If there are no valid indices, then we return 0.0
                if valid_idx.sum() == 0:
                    return 0.0
                # So, we only evaluate the loss on the indices where y_true is not nan
                # Also, we don't want to use the mean, since that will be affected by the number
                # of valid indices.  Instead, we use the sum, which will be the same regardless
                # of the number of valid indices.
                loss = torch.nn.MSELoss(reduction='sum')(y_computed[valid_idx], y_true[valid_idx])
                if torch.isnan(loss):
                    # At this point if the loss is still nan, then we raise an error, since 
                    # that means there is some case that our code is not handling.
                    raise ValueError("Loss is nan")
                return loss
            return loss_func
        else:
            raise ValueError(f"Unknown loss function {cfg['func']}")

    @staticmethod
    def initialization_func_factory(cfg):
        if cfg["func"] == "zeros":
            def initialization_func(device, rows=1):
                return torch.zeros((rows, cfg["size"]), device=device)
            return initialization_func
        else:
            raise ValueError(f"Unknown initialization function {cfg['func']}")

    @staticmethod
    def data_func_factory(cfg):
        if cfg["func"] == "insert":
            def data_func(z, x, 
                          flatten_input=cfg['flatten_input']):
                with torch.no_grad():
                    if flatten_input:
                        z[:, cfg["idx_list"]] = x.view(-1, len(cfg["idx_list"]))
                    else:
                        z[:, cfg["idx_list"]] = x
                    return z
            return data_func
        else:
            raise ValueError(f"Unknown data function {cfg['func']}")

    @staticmethod
    def output_func_factory(cfg):
        if cfg["func"] == "max":
            def output_func(z, sequence_idx, x_sequence, y_sequence, z_sequence):
                return torch.argmax(z[:, cfg['idx_list']], dim=1)
            return output_func
        elif cfg["func"] == "all":
            def output_func(z, sequence_idx, x_sequence, y_sequence, z_sequence):
                return z[:, cfg['idx_list']]
            return output_func
        elif cfg["func"] == "all_series":
            def output_func(z, sequence_idx, x_sequence, y_sequence, z_sequence):
                z_output_list = []
                for z in z_sequence:
                    z_output_list.append(z[:, cfg['idx_list']])
                return z_output_list
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
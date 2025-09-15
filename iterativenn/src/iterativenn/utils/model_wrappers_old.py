import torch

class DescriptionWithMNISTCELoss(object):
    def __init__(self, model):
        # The current state for the sequence
        self.model = model
        self.output_dim = len(self.model[0].y_index_list)
        self.hy = None
        self.iteration = None
        self.x_size = len(self.model[0].x_index_list)
        # this assumes position of x in model config should always be indexed 0. This is also required for compressed=True setting.
        assert self.model[0].x_index_list[0] == 0, "Invalid x_index_list, must start with 0"
        self.start_y_index = self.model[0].y_index_list[0] - len(self.model[0].x_index_list)
        self.hy_size = len(self.model[0].h_index_list) + len(self.model[0].y_index_list)


    def __call__(self, x, iteration: int):
        if iteration == 0: # iteration is a timestep
            # Initialize internal state
            self.hy = torch.zeros(1, self.hy_size, device=x.device)
            self.iteration = 0
        else:
            self.iteration += 1
            assert iteration == self.iteration, f'Called out of order {iteration} {self.iteration}'
        # if so, then proceed with call
        #   concatenate x into z
        #   call model to produce h and y
        #   remember h (and y if you want)
        #   return y

        # This should be somewhere else, like in the collate_fn?
        x = x.view(-1, self.x_size)
        # This is a touchy bit.  Other ways of doing this
        # lead to errors having to do with inplace
        # modifications to a tensor from which we compute
        # gradients.
        z = torch.cat((x, self.hy), dim=1)
        z_out = self.model(z)
        if z_out.shape[1] == z.shape[1]:
            print("WARNING: DescriptionWithMNISTCELoss is only defined for compressed models, but the given model appears to be square.")
            print("WARNING: This will likely lead to subtle training errors! Remove this assert if you disagree")
            assert False, "Square operator not allowed in DescriptionWithMNISTCELoss"
        self.hy = z_out[:, -self.hy_size:]  # todo fix for gym runs

        y = z_out[:, self.start_y_index:(self.start_y_index+self.output_dim)]
        return y

    def set_hy_size(self, add_dim_hy: int) -> None:
        """
        Use when we grow the bands iterative NN
        Args:
            add_dim_hy: add specified dimension to the input vector to iterative NN

        Returns:
            None
        """
        self.hy_size = self.hy_size + add_dim_hy

    def train_loss(self, y_true_list, y_list):
        # Do whatever processing you want to produce the loss
        # between a list of truths and a list of predictions
        # E.g., sum over all of the losses.
        loss_func = torch.nn.CrossEntropyLoss()
        loss = 0
        # NOTE: This depends on all of the losses in the sequence.  This
        # makes sense for a "description"
        for y_true, y in zip(y_true_list, y_list):
            loss += loss_func(y, torch.tensor([y_true], device=y.device))
        return loss

    def validation_loss(self, y_true_list, y_list):
        # Do whatever processing you want to produce the loss
        # between a list of truths and a list of predictions
        # E.g., return just the last loss.
        loss_func = torch.nn.CrossEntropyLoss()
        y_true = y_true_list[-1]
        y = y_list[-1]
        return loss_func(y, torch.tensor([y_true], device=y.device))

    def predict(self, y):
        return int(y.argmax())

class MLPWithMNISTCELoss(object):
    def __init__(self, model):
        # The current state for the sequence
        self.model = model
        self.output_dim = len(self.model[0].y_index_list)
        self.hy = None
        self.iteration = None
        self.x_size = len(self.model[0].x_index_list)
        # this assumes position of x in model config should always be indexed 0. This is also required for compressed=True setting.
        assert self.model[0].x_index_list[0] == 0, "Invalid x_index_list, must start with 0"
        self.start_y_index = self.model[0].y_index_list[0] - len(self.model[0].x_index_list)
        self.hy_size = len(self.model[0].h_index_list) + len(self.model[0].y_index_list)

    def __call__(self, x, iteration):
        if iteration == 0:
            # Initialize internal state
            self.hy = torch.zeros(1, self.hy_size, device=x.device)
            self.iteration = 0
        else:
            self.iteration += 1
            assert iteration == self.iteration, f'Called out of order {iteration} {self.iteration}'

        # This should be somewhere else, like in the collate_fn?
        x = x.view(-1, self.x_size)
        # if so, then proceed with call
        #   concatenate x into z
        #   call model to produce h and y
        #   remember h (and train_loss(self, y_true_list, y_list):bit.  Other ways of doing this
        # lead to errors having to do with inplace
        # modifications to a tensor from which we compute
        # gradients.
        z = torch.cat((x, self.hy), dim=1)

        z_out = self.model(z)
        if z_out.shape[1] == z.shape[1]:
            print("WARNING: MLPWithMNISTCELoss is only defined for compressed models, but the given model appears to be square.")
            print("WARNING: This will likely lead to subtle training errors! Remove this assert if you disagree")
            assert False, "Square operator not allowed in MLPWithMNISTCELoss"
        # self.hy.size vs 110 for mnist
        # self.hy = z_out[:, -self.hy_size:] # todo fix for gym runs

        y = z_out[:, self.start_y_index:(self.start_y_index+self.output_dim)]

        return y

    def set_hy_size(self, add_dim: int):
        self.hy_size = self.hy_size + add_dim

    def train_loss(self, y_true_list, y_list):
        # Do whatever processing you want to produce the loss
        # between a list of truths and a list of predictions
        # E.g., sum over all of the losses.
        loss_func = torch.nn.CrossEntropyLoss()
        # NOTE: This depends on only the last loss in the sequence.  This
        # makes sense for a "MLP"
        y_true = y_true_list[-1]
        y = y_list[-1]
        return loss_func(y, torch.tensor([y_true], device=y.device))

    def validation_loss(self, y_true_list, y_list):
        # Do whatever processing you want to produce the loss
        # between a list of truths and a list of predictions
        # E.g., return just the last loss.
        loss_func = torch.nn.CrossEntropyLoss()
        y_true = y_true_list[-1]
        y = y_list[-1]
        return loss_func(y, torch.tensor([y_true], device=y.device))

    def predict(self, y):
        return int(y.argmax())

class LSTMWithMNISTCELoss(object):
    def __init__(self, model):
        # The current state for the sequence
        self.model = model
        self.output_dim = self.model.proj_size
        self.x_size = self.model.input_size
        self.hidden_size = self.model.hidden_size

        self.hn =None
        self.cn = None
        self.iteration = None

    def __call__(self, x, iteration):
        if iteration == 0:
            # Initialize internal state
            #self.hy = torch.zeros(1, self.hy_size, device=x.device)
            self.iteration = 0
            # check if this is correct
            hi = torch.zeros(1, self.output_dim)
            ci = torch.zeros(1, self.hidden_size)
            self.hn = hi
            self.cn = ci
        else:
            self.iteration += 1
            assert iteration == self.iteration, f'Called out of order {iteration} {self.iteration}'

        # This should be somewhere else, like in the collate_fn?
        x = x.view(-1, self.x_size)
        # In recent versions of gymnasium the output type is float64, which may not
        # be compatible with the type of self.hn and self.cn
        x = x.to(self.hn.dtype)

        y, (hn, cn) = self.model(x, (self.hn, self.cn))
        self.hn = hn
        self.cn = cn
        # if so, then proceed with call
        #   concatenate x into z
        #   call model to produce h and y
        #   remember h (and train_loss(self, y_true_list, y_list):bit.  Other ways of doing this
        # lead to errors having to do with inplace
        # modifications to a tensor from which we compute
        # gradients.

        return y

    def train_loss(self, y_true_list, y_list):
        # Do whatever processing you want to produce the loss
        # between a list of truths and a list of predictions
        # E.g., sum over all of the losses.
        loss_func = torch.nn.CrossEntropyLoss()
        # NOTE: This depends on only the last loss in the sequence.  This
        # makes sense for a "MLP"
        y_true = y_true_list[-1]
        y = y_list[-1]
        return loss_func(y, torch.tensor([y_true], device=y.device))

    def validation_loss(self, y_true_list, y_list):
        # Do whatever processing you want to produce the loss
        # between a list of truths and a list of predictions
        # E.g., return just the last loss.
        loss_func = torch.nn.CrossEntropyLoss()
        y_true = y_true_list[-1]
        y = y_list[-1]
        return loss_func(y, torch.tensor([y_true], device=y.device))

    def predict(self, y):
        return int(y[-1, :].argmax())
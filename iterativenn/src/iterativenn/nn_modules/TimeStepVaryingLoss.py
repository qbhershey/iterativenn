import torch
import torch.nn as nn


class TimeStepVaryingLoss(nn.Module):
    """
    input_seq: sequence tensor
    target_seq: sequence tensor
    time_step_dict:
    {'mse': [start_time_step: int, end_time_step: int, weight: float],
     'l1_loss': [start_time_step: int, end_time_step: int, weight: float]}
    """

    def __init__(self):
        super().__init__()
        self.loss_mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss()
        self.t_loss = []

    def forward(self, input_seq, target_seq, time_step_dict):
        for loss_key, values in time_step_dict.items():
            a = values[0]
            b = values[1]
            with torch.no_grad():
                w = torch.tensor(values[2])
            if loss_key == "mse":
                self.t_loss.append(w * self.loss_mse(input_seq[a:b], target_seq[a:b]))
            elif loss_key == "cross-entropy":
                self.t_loss.append(w * self.cross_entropy(input_seq[a:b], target_seq[a:b]))
            elif loss_key == "l1-loss":
                self.t_loss.append(w * self.l1_loss(input_seq[a:b], target_seq[a:b]))
            else:
                raise NotImplementedError
        return sum(self.t_loss)
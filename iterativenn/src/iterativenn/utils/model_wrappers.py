import torch

class IterativeLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size=0, y_before_h=False):
        super(IterativeLSTM, self).__init__()
        self.model = torch.nn.LSTM(input_size, 
                                   hidden_size, 
                                   num_layers=1,
                                   proj_size=output_size)

        # The current state for the sequence
        self.x_size = self.model.input_size
        self.h_size = self.model.hidden_size

        # Note, if output_size == 0 then the output size is the same as the hidden size
        # but *not* using a linear layer to map the hidden state to the output
        if self.model.proj_size==0:
            self.y_size = self.model.hidden_size
        else:
            self.y_size = self.model.proj_size

        # If this is true, then this makes the order of z = (x, y, h) instead of (x, h, y)
        self.y_before_h = y_before_h

        # See the notes in 
        # notebooks/11-rcp-LSTM-sizes.ipynb
        # for more details on the sizes of the states.
        self.in_features = self.x_size+self.h_size+self.y_size
        self.out_features = self.in_features

    def __call__(self, z):
        x0 = z[:, 0:self.x_size]

        # Note, there is an unfortunate naming convention in the Pytorch LSTM implementation.
        # What we call "h" is actually the "cell state" in the LSTM notation, which is normally denoted "c".
        # What we call "y" is actually the "output state" or "hiddent state" in the LSTM notation, 
        # which is normally denoted "h".
        # There is a further complication that the "output state" is actually a learned projection of "h" so
        # its size is not the same as "c".
        # Accordingly, here we will stick with our definition of "h" and "y" and *not use* the LSTM notation

        # If this is true, then this makes the order of z = (x, y, h) instead of (x, h, y)
        if self.y_before_h:
            # Note, this is h0 in the LSTM notation, but we would call it part of y0
            y0 = z[:, self.x_size:self.x_size+self.y_size]
            # Note, this is c0 in the LSTM notation, but we would call it part of h0
            h0 = z[:, self.x_size+self.y_size:self.x_size+self.y_size+self.h_size]
        else:
            # Note, this is c0 in the LSTM notation, but we would call it part of h0
            h0 = z[:, self.x_size:self.x_size+self.h_size]
            # Note, this is h0 in the LSTM notation, but we would call it part of y0
            y0 = z[:, self.x_size+self.h_size:self.x_size+self.h_size+self.y_size]

        yn_1, (yn_2, hn) = self.model(x0.reshape(1, x0.shape[0], x0.shape[1]).contiguous(), 
                                      (y0.reshape(1, y0.shape[0], y0.shape[1]).contiguous(), # This is h0 in the LSTM notation
                                       h0.reshape(1, h0.shape[0], h0.shape[1]).contiguous())) # This is c0 in the LSTM notation

        # Note, if RCP understands the LSTM implementation, then
        # yn_1 and yn_2 are the same here since we our LSTM is
        # unidirectional and we have a sequence of length 1.        
        # Also, yn_1==yn_2 plays two roles.  It is part of the state for the next
        # iteration, but it is also the output of the LSTM for this iteration.
        assert torch.allclose(yn_1, yn_2), 'yn_1 and yn_2 are not the same, but they should be'

        # Now, we need to reshape yn and hn to be 2D tensors with shape N, H_out
        # which is what we want for the next iteration.
        yn = yn_1.reshape(-1, yn_1.shape[2])
        hn = hn.reshape(-1, hn.shape[2])

        # If this is true, then this makes the order of z = (x, y, h) instead of (x, h, y)
        if self.y_before_h:
            znew = torch.cat((x0, yn, hn), dim=1)
        else:
            znew = torch.cat((x0, hn, yn), dim=1)
        return znew
    
    def number_of_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class IterativeRNNLike(torch.nn.Module):
    def __init__(self, type, input_size, hidden_size):
        super(IterativeRNNLike, self).__init__()
        if type == 'RNN':
            self.model = torch.nn.RNN(input_size, 
                                      hidden_size,
                                      num_layers=1) 
        elif type == 'GRU':
            self.model = torch.nn.GRU(input_size, 
                                      hidden_size,
                                      num_layers=1)                                        
        else:
            raise ValueError('Unknown type: {}'.format(type))

        # The current state for the sequence
        self.x_size = self.model.input_size
        # Note, for the RNN and GRU, the hidden size is the same as the output size
        self.h_size = self.model.hidden_size
        self.y_size = self.model.hidden_size

        self.in_features = self.x_size+self.h_size+self.y_size
        self.out_features = self.in_features

    def __call__(self, z):
        x0 = z[:, 0:self.x_size]
        # Note, this is c0 in the LSTM notation, but we would call it part of h0
        h0 = z[:, self.x_size:self.x_size+self.h_size]

        # Note, x0 has shape N, H_in and h0 has shape N, H_out
        # where N is the batch size and H_in and H_out are the input and output
        # sizes of the RNN.  We need to reshape these to be 3D tensors with
        # shape 1, N, H_in and 1, N, H_out respectively.  
        # The initial "1" is the number of layers in the RNN, which is always 1
        # for us since we control the iterations of the RNN.
        # For details, see:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.RNN
        yn, hn = self.model(x0.reshape(1, x0.shape[0], x0.shape[1]).contiguous(),
                            h0.reshape(1, h0.shape[0], h0.shape[1]).contiguous())

        # Now, we need to reshape yn and hn to be 2D tensors with shape N, H_out
        # which is what we want for the next iteration.
        yn = yn.reshape(-1, self.model.hidden_size)
        hn = hn.reshape(-1, self.model.hidden_size)

        # Note, yh==hn plays two roles.  It is part of the state for the next
        # iteration, but it is also the output of the LSTM for this iteration.
        znew = torch.cat((x0, hn, yn), dim=1)
        return znew

    def number_of_trainable_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


class IterativeRNN(IterativeRNNLike):
    def __init__(self, input_size, hidden_size):
        super(IterativeRNN, self).__init__('RNN', input_size, hidden_size)

class IterativeGRU(IterativeRNNLike):
    def __init__(self, input_size, hidden_size):
        super(IterativeGRU, self).__init__('GRU', input_size, hidden_size)

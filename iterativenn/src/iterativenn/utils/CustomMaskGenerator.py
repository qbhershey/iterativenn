import torch
import numpy as np

class CustomMaskGenerator:
    """
    Args:
    input_mask: The mask to apply to the linear transformation.   Note, this
        can contain floating point numbers.  The mask is applied to the
        update so, if the mask contains a 0, then the update is set to 0
        and that entry is not updated by the gradient.
    dense: type float
    1 means mask is all 1's
    0 means mask is all 0's
    0.5 means half 1's and half 0's

    """
    def __init__(self, input_mask, prob, train=False):
        self.mask = input_mask
        self.prob = prob
        self.train = train



    def get_updated_mask(self):
        """
        This function assumes input masks is all 1's
        Currently on sprase version added which takes care of
        - all 1's
        - all 0's
        - random 0's and 1's

        TODO: add more mask updating schemes in this class.
        Returns:
        updated mask
        """
        self.mask = torch.nn.Dropout(self.prob)(self.mask)
        # undo the defualt scaling of dropout
        if self.mask.max()>1.0:
            self.mask /=self.mask.max()
        return self.mask
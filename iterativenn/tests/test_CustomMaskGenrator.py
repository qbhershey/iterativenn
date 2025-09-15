from iterativenn.utils.CustomMaskGenerator import CustomMaskGenerator
from iterativenn.nn_modules.MaskedLinear import MaskedLinear
import torch

class GetUpdateMask:
    def __init__(self):
        self.in_features = 20
        self.out_features = 30
        torch.manual_seed(0)
        self.m = MaskedLinear(self.in_features, self.out_features)


def test_get_updated_mask():
    tum = GetUpdateMask()
    cmg = CustomMaskGenerator(tum.m.mask, prob=0.0)
    mask = cmg.get_updated_mask()
    diff_is_small = torch.isclose(tum.m.mask, mask)
    assert torch.all(diff_is_small)
    del mask

    tum = GetUpdateMask()
    mask = CustomMaskGenerator(tum.m.mask, prob=1.0).get_updated_mask()
    diff_is_small = torch.isclose(torch.zeros_like(tum.m.mask), mask)
    assert torch.all(diff_is_small)

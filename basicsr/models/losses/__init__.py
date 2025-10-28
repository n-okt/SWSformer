# ------------------------------------------------------------------------
# Modified from CGNet (https://github.com/Ascend-Research/CascadedGaze)
# ------------------------------------------------------------------------
from .losses import (L1Loss, MSELoss, PSNRLoss, MultiHeadPSNRLoss)

__all__ = [
    'L1Loss', 'MSELoss', 'PSNRLoss', 'MultiHeadPSNRLoss',
]

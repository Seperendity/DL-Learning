import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type, Optional, Tuple

class PatchEmbed(nn.Moudle):
    def __init__(self, 
                 in_chans    : int = 3,
                 embed_dim   : int = 768,
                 kernel_size : int = 16,
                 padding     : int = 0,
                 stride      : int = 16,
                 ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    


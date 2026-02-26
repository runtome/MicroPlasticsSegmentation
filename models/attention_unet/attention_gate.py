"""
Attention Gate module for Attention U-Net.
Selectively amplifies relevant features in skip connections.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Additive attention gate.

    g: gating signal from decoder (coarser, semantically richer)
    x: skip connection from encoder (finer, spatially richer)
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        """
        Args:
            F_g: channels in gating signal
            F_l: channels in skip connection
            F_int: intermediate channels
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: gating signal (B, F_g, H', W')
            x: skip connection (B, F_l, H, W)
        Returns:
            Attention-weighted skip connection (B, F_l, H, W)
        """
        # Upsample g to match x spatial dimensions
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=True)
        g1 = self.W_g(g_up)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

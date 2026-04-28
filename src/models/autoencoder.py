"""
 * [INPUT]: 依赖 torch.nn, monai.networks.nets.SwinUNETR, src.models.weights
 * [OUTPUT]: 对外提供 Autoencoder3D 类
 * [POS]: src/models/ 的核心自编码器, 被训练与推理流程消费
 * [PROTOCOL]: 变更时更新此头部, 然后检查 CLAUDE.md
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from src.models.weights import load_swin_unetr_pretrained, freeze_encoder


# ============================================================
# Decoder Block
# ============================================================

class DecoderBlock(nn.Module):
    """
    3D U-Net 风格解码器块.
    上采样 + 卷积 + 跳跃连接融合.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.norm1(x)
        x = self.relu(x)
        # 对齐空间尺寸 (处理奇偶差异)
        if x.shape[2:] != skip.shape[2:]:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="trilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x


# ============================================================
# Autoencoder3D
# ============================================================

class Autoencoder3D(nn.Module):
    """
    3D U-Net 自编码器.
    编码器: MONAI SwinUNETR (可预训练/冻结).
    解码器: 自定义 4 层上采样 + 跳跃连接.
    重建头: Conv3d 映射到单通道.
    """

    def __init__(
        self,
        encoder_name: str = "swin_unetr",
        freeze_encoder: bool = True,
        feature_size: int = 48,
        use_checkpoint: bool = True,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        if encoder_name != "swin_unetr":
            raise ValueError(f"Unsupported encoder: {encoder_name}")

        # --------------------------------------------------
        # Encoder: SwinUNETR
        # --------------------------------------------------
        self.encoder = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            patch_size=2,
        )

        if pretrained:
            load_swin_unetr_pretrained(self.encoder, checkpoint_path)

        if freeze_encoder:
            from src.models.weights import freeze_encoder as _freeze_encoder
            _freeze_encoder(self.encoder)

        # --------------------------------------------------
        # Decoder: 4 layers
        # --------------------------------------------------
        # hidden_states_out shapes (feature_size=48):
        # [0] (B, 48, 32, 32, 32)
        # [1] (B, 96, 16, 16, 16)
        # [2] (B, 192, 8, 8, 8)
        # [3] (B, 384, 4, 4, 4)
        # [4] (B, 768, 2, 2, 2)  -- bottleneck
        self.decoder = nn.ModuleList([
            DecoderBlock(768, 384, 384),   # 2 -> 4
            DecoderBlock(384, 192, 192),   # 4 -> 8
            DecoderBlock(192, 96, 96),     # 8 -> 16
            DecoderBlock(96, 48, 48),      # 16 -> 32
        ])

        # --------------------------------------------------
        # Reconstruction Head
        # --------------------------------------------------
        self.reconstruction_head = nn.Sequential(
            nn.ConvTranspose3d(48, 48, kernel_size=2, stride=2),  # 32 -> 64
            nn.InstanceNorm3d(48),
            nn.ReLU(inplace=True),
            nn.Conv3d(48, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder forward with intermediate outputs
        hidden_states_out = self.encoder.swinViT(x, normalize=True)

        # Bottleneck is the last hidden state
        bottleneck = hidden_states_out[4]  # (B, 768, 2, 2, 2)

        # Decoder with skip connections
        d = bottleneck
        for i, dec_block in enumerate(self.decoder):
            skip = hidden_states_out[3 - i]
            d = dec_block(d, skip)

        # Reconstruction head
        out = self.reconstruction_head(d)
        return out

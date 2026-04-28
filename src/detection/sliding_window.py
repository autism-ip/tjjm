"""
 * [INPUT]: 依赖 torch, monai.inferers
 * [OUTPUT]: 对外提供 sliding_window_reconstruct
 * [POS]: src/detection/ 的整幅 CT 重建入口，负责 patch 切分与结果融合
 * [PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

import torch
from monai.inferers import SlidingWindowInferer


# ============================================================
# Sliding Window Reconstruction
# ============================================================

def sliding_window_reconstruct(
    model: torch.nn.Module,
    ct_scan: torch.Tensor,
    patch_size: tuple[int, int, int] = (64, 64, 64),
    stride: int = 32,
    batch_size: int = 4,
) -> torch.Tensor:
    """
    使用 MONAI SlidingWindowInferer 对整幅 CT 进行分块重建
    INPUT:  ct_scan (C, D, H, W)
    OUTPUT: reconstructed (C, D, H, W)
    """
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # (C, D, H, W) -> (B=1, C, H, W, D) 适配 MONAI 3D 格式
    if ct_scan.dim() == 4:
        ct_scan = ct_scan.permute(0, 2, 3, 1).unsqueeze(0)
    elif ct_scan.dim() == 3:
        ct_scan = ct_scan.permute(1, 2, 0).unsqueeze(0).unsqueeze(0)

    ct_scan = ct_scan.to(device)

    overlap_ratio = max(0.0, min(0.99, (patch_size[0] - stride) / patch_size[0]))
    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=batch_size,
        overlap=overlap_ratio,
        mode="gaussian",
    )

    with torch.no_grad():
        reconstructed = inferer(ct_scan, model)

    # (B=1, C, H, W, D) -> (C, D, H, W)
    if reconstructed.dim() == 5:
        reconstructed = reconstructed.squeeze(0).permute(0, 3, 1, 2)
    return reconstructed

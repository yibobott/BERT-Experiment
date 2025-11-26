import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertConfig

class RMSNorm(nn.Module):
    """
    RMSNorm: Root Mean Square Layer Normalization
    https://arxiv.org/abs/1910.07467
    A simple normalization scheme: normalize by the L2 root-mean-square of each
    token and then apply a learnable scaling parameter.
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # Compute root mean square (RMS): sqrt(mean(x^2)) along the last dim
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return self.weight * x_norm

def convert_layernorm_to_rmsnorm(module: nn.Module, prefix: str = "") -> nn.Module:
    """
    Recursively replace all nn.LayerNorm modules in `module` with RMSNorm.

    Note: the RMSNorm dimension is taken from the original LayerNorm's
    `normalized_shape[0]`.
    """
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.LayerNorm):
            # child.normalized_shape is a tuple, e.g. (768,)
            dim = child.normalized_shape[0]
            print(f"[RMSNorm] Replacing LayerNorm at: {child_prefix} (dim={dim})")
            setattr(module, name, RMSNorm(dim, eps=child.eps))
        else:
            convert_layernorm_to_rmsnorm(child, prefix=child_prefix)
    return module


class RMSNormBertForMaskedLM(BertForMaskedLM):
    """
    BERT-MLM model whose LayerNorms are replaced by RMSNorm,
    initialized from a config (no pretrained checkpoint).
    """

    def __init__(self, config: BertConfig):
        # 初始化原始 BertForMaskedLM 结构
        super().__init__(config)
        # 将 self 内部所有 LayerNorm 替换为 RMSNorm
        convert_layernorm_to_rmsnorm(self)
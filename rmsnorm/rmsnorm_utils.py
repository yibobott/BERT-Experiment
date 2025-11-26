import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertConfig

class RMSNorm(nn.Module):
    """
    RMSNorm: Root Mean Square Layer Normalization
    https://arxiv.org/abs/1910.07467
    实现非常简单，只根据每个 token 的 L2 均值归一化，然后乘以一个可学习的缩放参数。
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # 计算均方根 (RMS): sqrt(mean(x^2))
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return self.weight * x_norm

def convert_layernorm_to_rmsnorm(module: nn.Module, prefix: str = "") -> nn.Module:
    """
    递归地将模型中的所有 nn.LayerNorm 替换为 RMSNorm。
    注意：只会根据原 LayerNorm 的 normalized_shape[0] 来构造 RMSNorm。
    """
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.LayerNorm):
            # child.normalized_shape 是一个 tuple，比如 (768,)            
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
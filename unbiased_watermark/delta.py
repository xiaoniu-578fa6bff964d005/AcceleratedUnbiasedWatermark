from collections.abc import Callable
import numpy as np
import torch
from torch import FloatTensor, Tensor
from torch.nn import functional as F

from . import AbstractWatermarkCode, AbstractReweight


class Delta_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, u: FloatTensor):
        assert torch.all(u >= 0) and torch.all(u <= 1)
        self.u = u

    @classmethod
    def from_random_(
        cls,
        rng: np.ndarray,  # dtype=object, a nprandom.Generator
        vocab_size: int,
    ):
        u = np.empty(rng.shape, dtype=np.float32)
        for index in np.ndindex(rng.shape):
            u[index] = rng[index].random(dtype=np.float32)
        return cls(torch.tensor(u))

    def tensor_shape_map(
        self,
        shape_map: Callable[[Tensor], Tensor],
    ) -> "Delta_WatermarkCode":
        return self.__class__(shape_map(self.u))

    @classmethod
    def stack(
        cls,
        codes: list["Delta_WatermarkCode"],
        dim: int,
    ) -> "Delta_WatermarkCode":
        return cls(torch.stack([code.u for code in codes], dim=dim))

    @classmethod
    def concat(
        cls,
        codes: list["Delta_WatermarkCode"],
        dim: int,
    ) -> "Delta_WatermarkCode":
        return cls(torch.concat([code.u for code in codes], dim=dim))


class Delta_Reweight(AbstractReweight):
    watermark_code_type = Delta_WatermarkCode

    def __repr__(self):
        return f"Delta_Reweight()"

    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        assert isinstance(code, self.watermark_code_type)
        assert p_logits.shape[:-1] == code.u.shape
        cumsum = torch.cumsum(F.softmax(p_logits, dim=-1), dim=-1)
        index = torch.searchsorted(cumsum, code.u[..., None], right=True)
        index = torch.clamp(index, 0, p_logits.shape[-1] - 1)
        modified_logits = torch.where(
            torch.arange(p_logits.shape[-1], device=p_logits.device) == index,
            torch.full_like(p_logits, 0),
            torch.full_like(p_logits, float("-inf")),
        )
        return modified_logits

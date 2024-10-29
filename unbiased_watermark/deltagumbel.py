from collections.abc import Callable
import numpy as np
import torch
from torch import FloatTensor, Tensor
from torch.nn import functional as F

from . import AbstractWatermarkCode, AbstractReweight


def get_gumbel_variables(rng, vocab_size):
    u = rng.random((vocab_size,), dtype=np.float32)
    with np.errstate(divide="ignore"):
        e = -np.log(u)  # ~ Exp(1)
        g = -np.log(e)  # ~ Gumbel(0, 1)
    return u, e, g


class DeltaGumbel_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, g: FloatTensor):
        self.g = g

    @classmethod
    def from_random_(
        cls,
        rng: np.ndarray,  # dtype=object, a nprandom.Generator
        vocab_size: int,
    ):
        g = np.empty(rng.shape + (vocab_size,), dtype=np.float32)
        for index in np.ndindex(rng.shape):
            g[index] = get_gumbel_variables(rng[index], vocab_size)[2]
        return cls(torch.tensor(g))

    def tensor_shape_map(
        self,
        shape_map: Callable[[Tensor], Tensor],
    ):
        shape_map = torch.func.vmap(shape_map, in_dims=-1, out_dims=-1)
        return self.__class__(shape_map(self.g))

    @classmethod
    def stack(
        cls,
        codes: list["DeltaGumbel_WatermarkCode"],
        dim: int,
    ) -> "DeltaGumbel_WatermarkCode":
        if dim < 0:
            dim -= 1
        return cls(torch.stack([code.g for code in codes], dim=dim))

    @classmethod
    def concat(
        cls,
        codes: list["DeltaGumbel_WatermarkCode"],
        dim: int,
    ) -> "DeltaGumbel_WatermarkCode":
        if dim < 0:
            dim -= 1
        return cls(torch.concat([code.g for code in codes], dim=dim))


class DeltaGumbel_Reweight(AbstractReweight):
    watermark_code_type = DeltaGumbel_WatermarkCode

    def __repr__(self):
        return f"DeltaGumbel_Reweight()"

    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        assert isinstance(code, self.watermark_code_type)
        assert p_logits.shape == code.g.shape
        index = torch.argmax(p_logits + code.g, dim=-1)
        modified_logits = torch.where(
            torch.arange(p_logits.shape[-1], device=p_logits.device)
            == index.unsqueeze(-1),
            torch.full_like(p_logits, 0),
            torch.full_like(p_logits, float("-inf")),
        )
        return modified_logits

    #  def get_la_score(self, code):
    #      """likelihood agnostic score"""
    #      import math
    #
    #      return torch.tensor(math.log(2)) - torch.exp(-code.g)

import numpy as np
import torch
from torch import FloatTensor, LongTensor

from .. import DeltaGumbel_WatermarkCode, AbstractWatermarkCode
from . import TokenwiseAdditiveScore


class DeltaGumbel_X(TokenwiseAdditiveScore):
    watermark_code_type = DeltaGumbel_WatermarkCode

    @classmethod
    def from_watermarkcode(*args, **kwargs):
        raise ValueError(
            "Don't use from_watermarkcode for DeltaGumbel_X directly. Use DeltaGumbel_X.builder() instead."
        )

    @classmethod
    def builder(cls, Delta: float):
        return DeltaGumbel_X_Builder(Delta)

    @classmethod
    def score_from_watermarkcode(
        cls,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        p_logits: FloatTensor,
        q_logits: FloatTensor,
        floor: float,
        a1: float,
        a2: float,
    ) -> FloatTensor:
        assert isinstance(code, cls.watermark_code_type)
        assert code.g.shape[:-1] == ids.shape
        #  assert p_logits is None  # is likelihood agnostic score. not used here

        gs = torch.gather(code.g, -1, ids.unsqueeze(-1)).squeeze(-1)
        Us = torch.exp(-torch.exp(-gs))
        Xs = torch.log(floor * Us**a1 + Us**a2)
        return Xs

    def get_per_token_log_MGF(self, t: float) -> float:
        import scipy.integrate

        if isinstance(t, np.ndarray):
            assert t.size == 1
            t = t.item()

        def integrand(U, d):
            return (self.floor * U**self.a1 + U**self.a2) ** d

        return np.log(scipy.integrate.quad(integrand, 0, 1, args=(t,))[0])

    def get_per_token_mu(self) -> float:
        import scipy.integrate

        def integrand(U):
            return np.log(self.floor * U**self.a1 + U**self.a2)

        return scipy.integrate.quad(integrand, 0, 1)[0]


class DeltaGumbel_X_Builder:
    def __init__(self, Delta):
        floor = np.floor(1 / (1 - Delta))
        tildeDelta = (1 - Delta) * floor
        a1 = Delta / (1 - Delta)
        with np.errstate(divide="ignore"):
            # will happen when Delta = 1-1/floor
            a2 = tildeDelta / (1 - tildeDelta)
        self.Delta = Delta
        self.tildeDelta = tildeDelta
        self.floor = floor
        self.a1 = a1
        self.a2 = a2

    def from_watermarkcode(
        self,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        skipped: np.array,
        p_logits: FloatTensor,
        q_logits: FloatTensor,
    ) -> DeltaGumbel_X:
        assert ids.shape == skipped.shape
        s = DeltaGumbel_X(
            scores=DeltaGumbel_X.score_from_watermarkcode(
                code, ids, p_logits, q_logits, self.floor, self.a1, self.a2
            )
            .detach()
            .cpu()
            .numpy(),
            skipped=skipped,
        )
        s.Delta = self.Delta
        s.tildeDelta = self.tildeDelta
        s.floor = self.floor
        s.a1 = self.a1
        s.a2 = self.a2
        return s

    def __repr__(self):
        return f"DeltaGumbel_X(Delta={self.Delta})"

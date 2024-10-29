import numpy as np
import torch
from torch import FloatTensor, LongTensor

from .. import Gamma_WatermarkCode, AbstractWatermarkCode
from . import TokenwiseAdditiveScore


class Gamma_U(TokenwiseAdditiveScore):
    watermark_code_type = Gamma_WatermarkCode

    @classmethod
    def score_from_watermarkcode(
        cls,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        p_logits: FloatTensor = None,
        q_logits: FloatTensor = None,
    ) -> FloatTensor:
        assert isinstance(code, cls.watermark_code_type)
        assert code.shuffle.shape[:-1] == ids.shape
        #  assert p_logits is None  # is likelihood agnostic score. not used here

        vocab_size = code.shuffle.shape[-1]
        pos = torch.gather(code.unshuffle, -1, ids.unsqueeze(-1)).squeeze(-1)
        Us = (pos + 0.5) / vocab_size
        return Us

    @classmethod
    def from_watermarkcode(cls, code: AbstractWatermarkCode, *args, **kwargs):
        self = super().from_watermarkcode(code, *args, **kwargs)
        self.vocab_size = code.shuffle.shape[-1]
        return self

    def get_per_token_log_MGF(self, t: float) -> float:
        #  ref https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
        if t == 0:
            return 0
        x = -t / 2 / self.vocab_size - np.log(
            -self.vocab_size * np.expm1(-t / self.vocab_size)
        )
        if t < np.log(2):
            y = np.log(np.expm1(t))
        else:
            y = t + np.log1p(-np.exp(-t))
        return x + y

    def get_per_token_mu(self) -> float:
        return 0.5

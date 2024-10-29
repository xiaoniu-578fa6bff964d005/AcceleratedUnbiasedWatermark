import numpy as np
import torch
from torch import FloatTensor, LongTensor

from .. import DeltaGumbel_WatermarkCode, AbstractWatermarkCode
from . import TokenwiseAdditiveScore


class DeltaGumbel_C(TokenwiseAdditiveScore):
    watermark_code_type = DeltaGumbel_WatermarkCode

    @classmethod
    def score_from_watermarkcode(
        cls,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        p_logits: FloatTensor = None,
        q_logits: FloatTensor = None,
    ) -> FloatTensor:
        assert isinstance(code, cls.watermark_code_type)
        assert code.g.shape[:-1] == ids.shape
        #  assert p_logits is None  # is likelihood agnostic score. not used here

        gs = torch.gather(code.g, -1, ids.unsqueeze(-1)).squeeze(-1)
        Cs = -torch.exp(-gs)
        return Cs

    def get_per_token_log_MGF(self, t: float) -> float:
        return -np.log1p(t)

    def get_per_token_mu(self) -> float:
        return -1

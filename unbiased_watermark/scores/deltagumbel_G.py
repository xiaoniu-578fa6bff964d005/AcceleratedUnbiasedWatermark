import numpy as np
import torch
from torch import FloatTensor, LongTensor

from .. import DeltaGumbel_WatermarkCode, AbstractWatermarkCode
from . import TokenwiseAdditiveScore


class DeltaGumbel_G(TokenwiseAdditiveScore):
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
        return gs

    def get_per_token_log_MGF(self, t: float) -> float:
        import scipy.special

        if t >= 1:
            return np.inf
        return scipy.special.loggamma(1 - t)

    def get_log_p_value(self) -> float:
        import scipy.optimize

        s = self.get_mean_per_token_score()
        sol = scipy.optimize.minimize(
            lambda t: self.get_per_token_log_MGF(t) - t * s, 0.5, bounds=[(0, 1)]
        )
        return sol.fun

    def get_per_token_mu(self) -> float:
        return np.euler_gamma

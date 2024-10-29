import numpy as np
import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor

from .. import AbstractWatermarkCode
from . import TokenwiseAdditiveScore


@torch.no_grad()
def safe_minus(log_q: FloatTensor, log_p: FloatTensor) -> FloatTensor:
    llr = log_q - log_p
    llr.nan_to_num_(nan=0.0)
    return llr


class LLRScore(TokenwiseAdditiveScore):
    @classmethod
    def score_from_watermarkcode(
        cls,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        p_logits: FloatTensor = None,
        q_logits: FloatTensor = None,
    ) -> FloatTensor:
        assert p_logits is not None
        assert q_logits is not None
        assert p_logits.shape == q_logits.shape
        log_p = F.log_softmax(p_logits, dim=-1)
        log_q = F.log_softmax(q_logits, dim=-1)
        return safe_minus(log_q, log_p).gather(-1, ids.unsqueeze(-1)).squeeze(-1)

    def get_per_token_log_MGF(self, t: float) -> float:
        if t != 1:
            import warnings

            warnings.warn(
                "The MGF of LLR score is only available for t=1. Please avoid minimizing Chernoff bound w.r.t. t for LLR score."
            )
            # This is just a placeholder. The actual value is not available.
            return np.inf
        return 0

    def get_log_p_value(self) -> float:
        with np.errstate(over="ignore"):
            s = self.get_score()
        return -s

    def get_per_token_mu(self) -> float:
        # This is just a placeholder. The actual value is not available.
        return 0.0

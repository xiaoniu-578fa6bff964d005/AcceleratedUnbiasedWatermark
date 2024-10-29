from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch import FloatTensor, LongTensor, BoolTensor

from .. import AbstractWatermarkCode
from . import TokenwiseAdditiveScore
from .llr import safe_minus, LLRScore


@torch.no_grad()
def log_minus_exp(log_a: FloatTensor, log_b: FloatTensor) -> FloatTensor:
    return torch.where(
        log_a <= log_b + np.log(2),
        log_b + torch.log(torch.expm1(torch.clamp(safe_minus(log_a, log_b), min=0.0))),
        log_a + torch.log1p(-torch.exp(log_b - log_a)),
    )


@torch.no_grad()
def get_max_llr_v2(
    # shape = (..., vocab_size)
    log_p: FloatTensor,
    log_q: FloatTensor,
    batch_query: list[tuple[float, float]],  # log_dist_p, log_dist_q
):
    """
    require large memeory, but avoid loop in python.
    return max_llr: (query_size, ...)
    """

    # shape = (..., vocab_size)
    llr = safe_minus(log_q, log_p)
    # shape = (..., vocab_size)
    try:
        sort_index = torch.argsort(llr, dim=-1, descending=True)
    except torch.cuda.OutOfMemoryError as e:
        #  use cpu instead
        sort_index = torch.argsort(llr.cpu(), dim=-1, descending=True).to(llr.device)
    del llr
    # shape = (..., vocab_size)
    log_p = log_p.gather(-1, sort_index)
    log_q = log_q.gather(-1, sort_index)
    del sort_index

    # shape = (..., vocab_size)
    llr = safe_minus(log_q, log_p)

    # shape = (..., vocab_size)
    log_cumsum_p = torch.logcumsumexp(log_p, dim=-1)
    del log_p
    log_cumsum_q = torch.logcumsumexp(log_q, dim=-1)
    del log_q

    max_llrs = []
    for log_dist_p, log_dist_q in batch_query:
        # shape = (..., vocab_size)
        log_modified_cumsum_q = log_minus_exp(log_cumsum_q, log_dist_q)
        log_modified_cumsum_p = torch.logaddexp(
            log_cumsum_p,
            torch.tensor(
                log_dist_p, device=log_cumsum_p.device, dtype=log_cumsum_p.dtype
            ),
        )

        # shape = (..., vocab_size)
        modified_llr = safe_minus(log_modified_cumsum_q, log_modified_cumsum_p)
        del log_modified_cumsum_q
        del log_modified_cumsum_p

        # pad left modified_llr with -inf
        # shape = (..., vocab_size+1)
        modified_llr = F.pad(modified_llr, (1, 0), value=float("-inf"))
        #  get index by argmax
        # shape = (..., )
        cut_index = torch.where(
            torch.any(llr < modified_llr[..., :-1], dim=-1),
            torch.argmax((llr < modified_llr[..., :-1]).to(torch.int), dim=-1),
            torch.tensor(modified_llr.shape[-1] - 1, device=modified_llr.device),
        )
        # shape = (..., )
        max_llrs.append(modified_llr.gather(-1, cut_index.unsqueeze(-1)).squeeze(-1))
    # shape = (query_size, ...)
    max_llr = torch.stack(max_llrs, dim=0)
    del max_llrs
    return max_llr


class RobustLLRScore(LLRScore):
    def __init__(self, scores: np.ndarray, skipped: np.ndarray):
        #  assert scores.shape == skipped.shape
        self.scores = scores
        self.skipped = skipped

    @classmethod
    def from_watermarkcode(*args, **kwargs):
        raise ValueError(
            "Don't use from_watermarkcode for RobustLLRScore directly. Use RobustLLRScore.builder() instead."
        )

    @classmethod
    def builder(cls, batch_query: list[tuple[float, float]]):
        return RobustLLRScoreBuilder(batch_query)

    @classmethod
    def score_from_watermarkcode(
        cls,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        p_logits: FloatTensor,
        q_logits: FloatTensor,
        batch_query: list[tuple[float, float]],
    ) -> FloatTensor:
        """
        ids: [...]
        p_logits: [..., vocab_size]
        return: robust_llr: [query_size, ...]
        """
        assert p_logits.shape == q_logits.shape
        log_p = F.log_softmax(p_logits, dim=-1)
        log_q = F.log_softmax(q_logits, dim=-1)
        max_llr = get_max_llr_v2(log_p, log_q, batch_query)
        min_llr = -get_max_llr_v2(log_q, log_p, [(q, p) for p, q in batch_query])
        trivial_pos = max_llr < min_llr
        max_llr = torch.where(
            trivial_pos, torch.tensor(0.0, device=max_llr.device), max_llr
        )
        min_llr = torch.where(
            trivial_pos, torch.tensor(0.0, device=min_llr.device), min_llr
        )
        llr = safe_minus(log_q, log_p).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        return llr.unsqueeze(0).clamp(min=min_llr, max=max_llr)

    def get_score(self) -> float:
        with np.errstate(over="ignore"):
            scores = (
                (self.scores * np.expand_dims(~self.skipped, 0))
                .reshape(self.scores.shape[0], -1)
                .sum(axis=1)
            )
        return scores.max()

    def best_query(self) -> tuple[float, float]:
        scores = (
            (self.scores * np.expand_dims(~self.skipped, 0))
            .reshape(self.scores.shape[0], -1)
            .sum(axis=1)
        )
        best_query_index = scores.argmax()
        return self.batch_query[best_query_index]

    def get_log_p_value(self) -> float:
        ns = -self.get_score()
        return ns + np.log(len(self.batch_query), dtype=ns.dtype)

    def get_per_token_mu(self) -> float:
        # This is just a placeholder. The actual value is not available.
        return 0.0


class RobustLLRScoreBuilder:
    def __init__(self, batch_query: list[tuple[float, float]]):
        self.batch_query = batch_query

    def from_watermarkcode(
        self,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        skipped: np.array,
        p_logits: FloatTensor,
        q_logits: FloatTensor,
    ) -> RobustLLRScore:
        assert ids.shape == skipped.shape
        assert p_logits.shape[:-1] == ids.shape
        assert q_logits.shape[:-1] == ids.shape
        s = RobustLLRScore(
            scores=RobustLLRScore.score_from_watermarkcode(
                code, ids, p_logits, q_logits, self.batch_query
            )
            .detach()
            .cpu()
            .numpy(),
            skipped=skipped,
        )
        s.batch_query = self.batch_query
        return s

    def __repr__(self):
        bq = self.batch_query
        if isinstance(bq, np.ndarray):
            bq = bq.tolist()
        s = ", ".join(f"({p:.2f}, {q:.2f})" for p, q in bq[:3])
        if len(bq) > 3:
            s += ", ..."
        return f"RobustLLRScoreBuilder(batch_query={s})"

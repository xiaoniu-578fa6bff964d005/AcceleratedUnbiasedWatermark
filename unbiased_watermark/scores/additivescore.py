from abc import ABC, abstractmethod

import numpy as np
from torch import FloatTensor, LongTensor, BoolTensor

from .. import AbstractWatermarkCode
from . import AbstractScore


class AbstractAdditiveScore(AbstractScore):
    def __add__(self, other: "AbstractAdditiveScore") -> "AbstractAdditiveScore":
        return AddScore(self, other)

    @abstractmethod
    def get_log_MGF(self, t: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_score(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_mu(self) -> float:
        raise NotImplementedError

    def get_log_p_value(self) -> float:
        import scipy.optimize

        s = self.get_score()
        sol = scipy.optimize.minimize(
            lambda t: self.get_log_MGF(t) - t * s, 1, bounds=[(0, None)]
        )
        return sol.fun


class AddScore(AbstractAdditiveScore):
    def __init__(self, score1: AbstractAdditiveScore, score2: AbstractAdditiveScore):
        self.score1 = score1
        self.score2 = score2

    def get_log_MGF(self, t: float) -> float:
        return self.score1.get_log_MGF(t) + self.score2.get_log_MGF(t)


class TokenwiseAdditiveScore(AbstractAdditiveScore):
    def __init__(self, scores: np.ndarray, skipped: np.ndarray):
        assert scores.shape == skipped.shape
        self.scores = scores
        self.skipped = skipped

    def __add__(self, other: "TokenwiseAdditiveScore") -> "TokenwiseAdditiveScore":
        assert self.scores.shape[-1] == other.scores.shape[-1]
        return self.__class__(
            scores=np.concatenate([self.scores, other.scores], axis=-1),
            skipped=np.concatenate([self.skipped, other.skipped], axis=-1),
        )

    def get_num_added(self) -> int:
        return self.skipped.size - self.skipped.sum()

    @abstractmethod
    def get_per_token_log_MGF(self, t: float) -> float:
        raise NotImplementedError

    def get_log_MGF(self, t: float) -> float:
        return self.get_per_token_log_MGF(t) * self.get_num_added()

    def get_score(self) -> float:
        return (self.scores * (~self.skipped)).sum()

    def get_mean_per_token_score(self) -> float:
        return self.get_score() / self.get_num_added()

    def get_log_p_value(self) -> float:
        import scipy.optimize

        s = self.get_mean_per_token_score()
        sol = scipy.optimize.minimize(
            lambda t: self.get_per_token_log_MGF(t) - t * s, 1, bounds=[(0, None)]
        )
        if sol.fun is np.nan:
            raise ValueError(
                "Invalid log p-value encountered while optimizing Chernoff bound"
            )
        return sol.fun * self.get_num_added()

    @classmethod
    @abstractmethod
    def score_from_watermarkcode(
        cls,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        p_logits: FloatTensor = None,
        q_logits: FloatTensor = None,
    ) -> FloatTensor:
        raise NotImplementedError

    @classmethod
    def from_watermarkcode(
        cls,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        skipped: np.array,
        p_logits: FloatTensor = None,
        q_logits: FloatTensor = None,
    ) -> AbstractScore:
        assert ids.shape == skipped.shape
        if p_logits is not None:
            assert p_logits.shape[:-1] == ids.shape
        if q_logits is not None:
            assert q_logits.shape[:-1] == ids.shape
        return cls(
            scores=cls.score_from_watermarkcode(code, ids, p_logits, q_logits)
            .detach()
            .cpu()
            .numpy(),
            skipped=skipped,
        )

    @abstractmethod
    def get_per_token_mu(self) -> float:
        raise NotImplementedError

    def get_mu(self) -> float:
        return self.get_per_token_mu() * self.get_num_added()

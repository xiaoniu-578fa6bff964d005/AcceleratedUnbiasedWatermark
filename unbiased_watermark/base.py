from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np
import torch
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import functional as F


class AbstractWatermarkCode(ABC):
    @classmethod
    @abstractmethod
    def from_random_(
        cls,
        rng: np.ndarray,  # dtype=object, a nprandom.Generator
        vocab_size: int,
    ):
        raise NotImplementedError

    @classmethod
    def from_random(
        cls,
        rng: np.random.Generator | list[np.random.Generator] | np.ndarray,
        vocab_size: int,
    ):
        """When rng is a list, it should have the same length as the batch size."""
        if isinstance(rng, np.random.Generator):
            rng = np.full((), rng, dtype=object)
        if isinstance(rng, list):
            rng = np.array(rng, dtype=object)
        return cls.from_random_(
            rng=rng,
            vocab_size=vocab_size,
        )

    @abstractmethod
    def tensor_shape_map(
        self,
        shape_map: Callable[[Tensor], Tensor],
    ) -> "AbstractWatermarkCode":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def stack(
        cls,
        codes: list["AbstractWatermarkCode"],
        dim: int,
    ) -> "AbstractWatermarkCode":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concat(
        cls,
        codes: list["AbstractWatermarkCode"],
        dim: int,
    ) -> "AbstractWatermarkCode":
        raise NotImplementedError


class AbstractReweight(ABC):
    watermark_code_type: type[AbstractWatermarkCode]

    @abstractmethod
    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        pass


#  class AbstractScore(ABC):
#      @abstractmethod
#      def get_log_p_value(self) -> float:
#          pass


#  class AbstractScore(ABC):
#      @abstractmethod
#      def score(self, p_logits: FloatTensor, q_logits: FloatTensor) -> FloatTensor:
#          """p is the original distribution, q is the distribution after reweighting."""
#          pass
#
#
#  class LLR_Score(AbstractScore):
#      def score(self, p_logits: FloatTensor, q_logits: FloatTensor) -> FloatTensor:
#          return F.log_softmax(q_logits, dim=-1) - F.log_softmax(p_logits, dim=-1)
#
#
#  class AbstractContextCodeExtractor(ABC):
#      @abstractmethod
#      def extract(self, context: LongTensor) -> any:
#          """Should return a context code `c` which will be used to initialize a torch.Generator."""
#          pass

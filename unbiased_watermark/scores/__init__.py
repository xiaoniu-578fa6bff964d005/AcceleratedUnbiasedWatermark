from abc import ABC, abstractmethod

from torch import FloatTensor, LongTensor, BoolTensor

from .. import AbstractWatermarkCode


class AbstractScore(ABC):
    @abstractmethod
    def get_log_p_value(self) -> float:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_watermarkcode(
        cls,
        code: AbstractWatermarkCode,
        ids: LongTensor,
        skipped: BoolTensor,
        p_logits: FloatTensor = None,  # likelihood based score
        q_logits: FloatTensor = None,
    ) -> "AbstractScore":
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}"


from .additivescore import AbstractAdditiveScore, TokenwiseAdditiveScore
from .deltagumbel_U import *
from .deltagumbel_G import *
from .deltagumbel_C import *
from .deltagumbel_A import *
from .deltagumbel_X import *
from .gamma_U import *
from .llr import *
from .robust_llr import RobustLLRScore

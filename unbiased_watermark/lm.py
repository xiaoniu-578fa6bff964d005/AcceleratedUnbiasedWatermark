from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import LongTensor, FloatTensor, BoolTensor
import torch.nn.functional as F

from . import AbstractWatermarkCode, AbstractReweight
from .scores import AbstractScore


class AbstractContextCodeExtractor(ABC):
    @abstractmethod
    def extract(self, context: LongTensor) -> np.ndarray:
        """
        Should return a context code `c` which will be used to initialize a torch.Generator.
        :param context: (..., seq_len)
        :return: (..., ), np.ndarray, dtype=obj or any
        """
        pass


def peel_ndarray(a, last_n_dim=1):
    ra = np.empty(a.shape[:-last_n_dim], dtype=object)
    for index in np.ndindex(ra.shape):
        ra[index] = a[index].copy()
    return ra


@dataclass(frozen=True)
class PrevN_ContextCodeExtractor(AbstractContextCodeExtractor):
    """Extracts the last n tokens in the context"""

    n: int

    def extract(self, context: LongTensor) -> np.ndarray:
        c = context[..., -self.n :].detach().cpu().numpy()
        c = peel_ndarray(c, last_n_dim=1)
        c = np.vectorize(lambda x: x.tobytes())(c)
        return c


class ContextCodeHistory:
    def __init__(self, data: np.ndarray = None, batch_shape=()):
        """
        data: (..., ), np.ndarray, dtype=obj, each element is a list
        """
        if data is None:
            data = np.empty(batch_shape, dtype=object)
            data.fill([])
        self.data = data

    def get_flattened(self) -> set:
        """
        :return: set of context code
        """
        return set(
            cc
            for cch_list in np.nditer(self.data, flags=["refs_ok"])
            for cc in cch_list.item()
        )

    def step(
        self, cc_extractor: AbstractContextCodeExtractor, context: LongTensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        :param context: (..., seq_len)
        :return: context_code, skipped
        context_code: (..., ), np.ndarray, dtype=obj
        skipped: (..., ), torch.bool
        """
        fcch = self.get_flattened()
        cc = cc_extractor.extract(context)
        skipped = np.zeros(context.shape[:-1], dtype=bool)
        for cc_item, skipped_item in np.nditer(
            [cc, skipped], op_flags=[["readwrite"], ["writeonly"]]
        ):
            cc_item = cc_item.item()
            if cc_item not in fcch:
                fcch.add(cc_item)
            else:
                skipped_item[()] = True
        self.add_context_code(cc)
        return cc, skipped

    def add_context_code(self, context_code: np.ndarray):
        """
        :param context_code: (..., ), np.ndarray, dtype=obj
        """
        assert context_code.shape == self.data.shape
        for index in np.ndindex(self.data.shape):
            self.data[index].append(context_code[index])

    def rollback(self, n: int):
        assert n >= 0
        if n == 0:
            return
        for index in np.ndindex(self.data.shape):
            self.data[index] = self.data[index][:-n]


def get_rng(*bs: bytes) -> torch.Generator:
    import hashlib

    m = hashlib.sha256()
    for b in bs:
        m.update(b)
    full_hash = m.digest()
    seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
    return np.random.default_rng(seed)


def step_watermark(
    reweight: AbstractReweight,
    p_logits: FloatTensor,
    input_ids: LongTensor,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
) -> tuple[FloatTensor, np.ndarray, AbstractWatermarkCode, np.ndarray]:
    """
    :param p_logits: (..., vocab_size)
    :param input_ids: (..., seq_len)
    :param cc_extractor: AbstractContextCodeExtractor
    :param cch: ContextCodeHistory, will be updated
    :return: log_q, context_code, watermark_code, skipped
    log_q: (..., vocab_size)
    context_code: (..., ), np.ndarray, dtype=obj
    watermark_code: AbstractWatermarkCode, shape: (..., )
    skipped: (..., )
    """
    cc, skipped = cch.step(cc_extractor, input_ids)
    rng = np.empty(cc.shape, dtype=object)
    for index in np.ndindex(rng.shape):
        rng[index] = get_rng(cc[index], private_key)
    watermark_code_type = reweight.watermark_code_type
    watermark_code = reweight.watermark_code_type.from_random(rng, p_logits.size(-1))
    watermark_code = watermark_code.tensor_shape_map(lambda x: x.to(input_ids.device))
    q_logits = reweight.reweight_logits(watermark_code, p_logits)
    #  if skipped then log_p otherwise log_q
    pytorch_skipped = torch.tensor(skipped, dtype=torch.bool, device=input_ids.device)
    wm_logits = torch.where(pytorch_skipped.unsqueeze(-1), p_logits, q_logits)
    return wm_logits, q_logits, cc, watermark_code, skipped


def detect_pre(
    vocab_size: int,
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
    out_ids: LongTensor,
    in_ids: LongTensor = None,
    p_logits: FloatTensor = None,
) -> tuple[FloatTensor, np.ndarray, AbstractWatermarkCode, np.ndarray]:
    """
    :param reweight: AbstractReweight
    :param cc_extractor: AbstractContextCodeExtractor
    :param cch: ContextCodeHistory
    :param out_ids: (..., out_seq_len)
    :param p_logits: (..., out_seq_len, vocab_size)
    :param in_ids: (..., in_seq_len)
    :return: log_q, context_code, watermark_code, skipped
    log_q: (..., out_seq_len, vocab_size)
    context_code: (..., out_seq_len), np.ndarray, dtype=obj
    watermark_code: AbstractWatermarkCode, shape: (..., out_seq_len)
    skipped: (..., out_seq_len)
    """
    batch_shape = out_ids.shape[:-1]
    assert cch.data.shape == batch_shape
    if in_ids is not None:
        assert in_ids.shape[:-1] == batch_shape
    if p_logits is not None:
        assert p_logits.shape[:-2] == batch_shape

    ids = out_ids if in_ids is None else torch.cat([in_ids, out_ids], dim=-1)
    cc_s, skipped_s = [], []
    for i in range(ids.shape[-1] - out_ids.shape[-1], ids.shape[-1]):
        cc, skipped = cch.step(cc_extractor, ids[..., :i])
        cc_s.append(cc)
        skipped_s.append(skipped)
    cc = np.stack(cc_s, axis=-1)
    skipped = np.stack(skipped_s, axis=-1)
    rng = np.empty(cc.shape, dtype=object)
    for index in np.ndindex(rng.shape):
        rng[index] = get_rng(cc[index], private_key)
    watermark_code_type = reweight.watermark_code_type
    watermark_code = reweight.watermark_code_type.from_random(rng, vocab_size)
    watermark_code = watermark_code.tensor_shape_map(lambda x: x.to(out_ids.device))
    if p_logits is not None:
        q_logits = reweight.reweight_logits(watermark_code, p_logits)
    else:
        q_logits = None
    pytorch_skipped = torch.tensor(skipped, dtype=torch.bool, device=out_ids.device)
    if p_logits is not None and p_logits.shape[-2] == out_ids.shape[-1]:
        wm_logits = torch.where(pytorch_skipped.unsqueeze(-1), p_logits, q_logits)
    else:
        wm_logits = None
    return wm_logits, q_logits, cc, watermark_code, skipped


def detect(
    vocab_size: int,
    score_type: type[AbstractScore],
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
    out_ids: LongTensor,
    in_ids: LongTensor = None,
    p_logits: FloatTensor = None,
) -> AbstractScore:
    wm_logits, q_logits, cc, watermark_code, skipped = detect_pre(
        vocab_size, reweight, cc_extractor, cch, private_key, out_ids, in_ids, p_logits
    )
    score = score_type.from_watermarkcode(
        watermark_code,
        out_ids,
        skipped=skipped,
        p_logits=p_logits,
        q_logits=q_logits,
    )
    return score

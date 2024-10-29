from .utils import *
from .basic_watermark import gen_n_token_uwm
from .mc import gen_mc


@torch.no_grad()
#  @profile_each_line
def gen_mc_last_uwm(
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
    ref_context_code: np.ndarray,
    ref_watermark_code: uwm.AbstractWatermarkCode,
    ref_skipped: np.ndarray,
    model,
    input_ids: LongTensor,
    ref_output_ids: LongTensor,
    ref_logprobs: FloatTensor,
    only_last: bool,
    past_key_values=None,
    process_logits_kwargs={},
) -> tuple[LongTensor, FloatTensor, FloatTensor, any, bool]:
    """
    reweight:
    cc_extractor:
    cch: (batch_size, )
    private_key:
    ref_context_code: (batch_size, n)
    ref_watermark_code: (batch_size, n)
    ref_skipped: (batch_size, n)
    model: Decoder-only model
    input_ids: (batch_size, seq_len). batch_size must be 1
    ref_output_ids: (batch_size, n)
    ref_logprobs: (batch_size, n, vocab_size)
    only_last: bool.  If True, only reweight the last token. If False, reweight all tokens.
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
    return: (context_code, watermark_code, skipped, output_ids, output_logprobs, watermark_logprobs, poverlaps, past_key_values, got_eos)
    context_code: (batch_size, gen_len)
    watermark_code: (batch_size, gen_len)
    skipped: (batch_size, gen_len)
    output_ids: (batch_size, gen_len)
    output_logprobs: (batch_size, gen_len, vocab_size)
    watermark_logprobs: (batch_size, gen_len, vocab_size)
    poverlaps: (batch_size, min(gen_len,n))
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one token in output_ids
    got_eos: bool
    """
    assert input_ids.shape[0] == 1
    assert ref_output_ids.shape == ref_logprobs.shape[:-1]
    #  get ground truth logprobs
    if past_key_values is not None:
        # shape (batch_size, num_heads, n-1, head_dim)
        cached_n = past_key_values[0][0].shape[2]
        input_tokens = torch.cat([input_ids[:, cached_n:], ref_output_ids], dim=1)
    else:
        input_tokens = torch.cat([input_ids, ref_output_ids], dim=1)
    _ids = torch.cat([input_ids, ref_output_ids], dim=1)
    # _ids: (batch_size, seq_len+n)
    output = model(input_tokens, past_key_values=past_key_values)
    #  logits = output.logits.clone()
    logits = output.logits
    logits = logits[:, -ref_output_ids.shape[1] - 1 :, :]
    if process_logits_kwargs != {}:
        #  for i in range(input_tokens.shape[1]):
        #      logits[:, i, :] = process_logits(
        #          _ids[:, : _ids.shape[1] - input_tokens.shape[1] + i + 1],
        #          logits[:, i, :],
        #          **process_logits_kwargs,
        #      )
        for i in range(-1, ref_output_ids.shape[1]):
            logits[:, i + 1, :] = process_logits(
                _ids[:, : _ids.shape[1] - ref_output_ids.shape[1] + i + 1],
                logits[:, i + 1, :],
                **process_logits_kwargs,
            )
    # logits: (batch_size, n+1, vocab_size)
    if not only_last:
        q_logits = reweight.reweight_logits(ref_watermark_code, logits[:, :-1, :])
        pytorch_ref_skipped = torch.tensor(
            ref_skipped, dtype=torch.bool, device=q_logits.device
        )
        target_logits = torch.where(
            pytorch_ref_skipped.unsqueeze(-1),
            logits[:, :-1, :],
            q_logits,
        )
    else:
        target_logits = logits[:, :-1, :]
    # target_logprob: (batch_size, n, vocab_size)

    gen_tokens, watermarked_logprobs, poverlaps, fully_coupled = mc_sample(
        target_logits[0, :, :],  # shape (n, vocab_size)
        ref_logprobs[0],
        ref_output_ids[0],
    )
    # gen_tokens: (min(gen_len,n))
    # watermarked_logprobs: (min(gen_len,n), vocab_size)
    # poverlaps: (min(gen_len,n))
    got_eos = False
    if gen_tokens[-1] == model.config.eos_token_id:
        got_eos = True
    if fully_coupled and not got_eos:
        (
            last_watermarked_logits,
            _last_q_logits,
            last_cc,
            last_watermark_code,
            last_skipped,
        ) = step_watermark(
            reweight, logits[:, -1, :], _ids, cc_extractor, cch, private_key
        )
        # last_watermarked_logits: (batch_size, vocab_size)
        # last_cc: (batch_size, )
        # last_watermark_code: (batch_size, )
        # last_skipped: (batch_size, )
        cc = np.concatenate([ref_context_code, last_cc[:, None]], axis=1)
        watermark_code = ref_watermark_code.concat(
            [
                ref_watermark_code,
                last_watermark_code.tensor_shape_map(lambda x: x.unsqueeze(-1)),
            ],
            dim=-1,
        )
        skipped = np.concatenate([ref_skipped, last_skipped[:, None]], axis=1)
        new_token, last_watermarked_logprobs = basic_sample(last_watermarked_logits)
        # last_watermarked_logprobs: (batch_size, vocab_size)
        watermarked_logprobs = torch.cat(
            [
                watermarked_logprobs.unsqueeze(0),
                last_watermarked_logprobs.unsqueeze(1),
            ],
            dim=1,
        )
        # watermarked_logprobs: (batch_size, n+1, vocab_size)
        # new_token: (batch_size=1, 1)
        output_ids = torch.cat([gen_tokens.unsqueeze(0), new_token], dim=-1)
        # output_ids: (1, n+1)
        output_logprobs = F.log_softmax(logits, dim=-1)
        # output_logprobs: (1, n+1, vocab_size)
        if (new_token == model.config.eos_token_id).all():
            got_eos = True
    else:
        gen_len = gen_tokens.shape[0]
        output_ids = gen_tokens.unsqueeze(0)
        # output_ids: (1, gen_len)
        output_logprobs = F.log_softmax(logits[:, :gen_len, :], dim=-1)
        # output_logprobs: (1, gen_len, vocab_size)
        cch.rollback(ref_output_ids.shape[1] - gen_len)
        cc = ref_context_code[:, :gen_len]
        watermark_code = ref_watermark_code.tensor_shape_map(lambda x: x[:, :gen_len])
        skipped = ref_skipped[:, :gen_len]
    poverlaps = poverlaps.unsqueeze(0)
    # poverlaps: (1, gen_len)

    # fix past_key_values
    past_key_values = output.past_key_values
    # each tensor is of shape (batch_size, num_heads, sequence_length, embed_size_per_head)
    past_key_values = tree_map(
        lambda x: x[:, :, : input_ids.shape[1] + output_ids.shape[1] - 1],
        past_key_values,
    )

    return (
        cc,
        watermark_code,
        skipped,
        output_ids,
        output_logprobs,
        watermarked_logprobs,
        poverlaps,
        past_key_values,
        got_eos,
    )


#  @profile_each_line
def mc_uwm_sample_generator(
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
    reweight_in_mc: bool,
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    past_key_values=None,
    ref_past_key_values=None,
    **kwargs,
):
    model.eval()
    ref_model.eval()
    while True:
        (
            ref_context_code,
            ref_watermark_code,
            ref_skipped,
            ref_output_ids,
            ref_logprobs,
            ref_watermark_logprobs,
            ref_past_key_values,
            _got_eos,
        ) = gen_n_token_uwm(
            reweight,
            cc_extractor,
            cch,
            private_key,
            ref_model,
            input_ids,
            n,
            past_key_values=ref_past_key_values,
            **kwargs,
        )
        if reweight_in_mc:
            (
                cc,
                watermark_code,
                skipped,
                output_ids,
                output_logprobs,
                watermark_logprobs,
                poverlaps,
                past_key_values,
                got_eos,
            ) = gen_mc_last_uwm(
                reweight,
                cc_extractor,
                cch,
                private_key,
                ref_context_code,
                ref_watermark_code,
                ref_skipped,
                model,
                input_ids,
                ref_output_ids,
                ref_watermark_logprobs,  # not ref_logprobs
                only_last=False,
                past_key_values=past_key_values,
                **kwargs,
            )
        else:
            (
                cc,
                watermark_code,
                skipped,
                output_ids,
                output_logprobs,
                watermark_logprobs,
                poverlaps,
                past_key_values,
                got_eos,
            ) = gen_mc_last_uwm(
                reweight,
                cc_extractor,
                cch,
                private_key,
                ref_context_code,
                ref_watermark_code,
                ref_skipped,
                model,
                input_ids,
                ref_output_ids,
                ref_logprobs,  # not ref_watermark_logprobs
                only_last=True,
                past_key_values=past_key_values,
                **kwargs,
            )
        ref_past_key_values = fix_gen_n_token_pass_key_values(
            ref_output_ids, output_ids, ref_past_key_values
        )
        yield output_ids, output_logprobs
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        if got_eos:
            break

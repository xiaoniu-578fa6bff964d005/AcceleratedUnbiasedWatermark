from .utils import *


@torch.no_grad()
def gen_n_token_uwm(
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
    model,
    input_ids: LongTensor,
    n: int,
    past_key_values=None,
    process_logits_kwargs={},
) -> tuple[
    np.ndarray,
    AbstractWatermarkCode,
    np.ndarray,
    LongTensor,
    FloatTensor,
    FloatTensor,
    any,
    bool,
]:
    """
    reweight:
    cc_extractor:
    cch: (batch_size, )
    private_key:
    model: Decoder-only model
    input_ids: (batch_size, seq_len), need to be on the same device and appropriate dtype
    n: number of tokens to generate
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
    return: (context_code, watermark_code, skipped, output_ids, output_logprobs, watermark_logprobs, past_key_values, got_eos)
    context_code: (batch_size, n)
    watermark_code: (batch_size, n)
    skipped: (batch_size, n)
    output_ids: (batch_size, n)
    output_logprobs: (batch_size, n, vocab_size)
    watermark_logprobs: (batch_size, n, vocab_size)
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one token in output_ids
    got_eos: bool
    """
    assert cch.data.shape == input_ids.shape[:-1]
    if past_key_values is not None:
        # shape (batch_size, num_heads, n-1, head_dim)
        cached_n = past_key_values[0][0].shape[2]
        input_tokens = input_ids[:, cached_n:]
    else:
        input_tokens = input_ids
    output_ids = []
    output_logprobs = []
    watermarked_logprobs = []
    ccs = []
    watermark_codes = []
    skippeds = []
    device = model.device
    got_eos = False
    for i in range(n):
        output = model(
            input_tokens,
            past_key_values=past_key_values,
        )
        logits = output.logits[:, -1, :]
        logits = process_logits(input_ids, logits, **process_logits_kwargs)
        logprobs = F.log_softmax(logits, dim=-1)
        wm_logits, _q_logts, cc, watermark_code, skipped = step_watermark(
            reweight, logits, input_ids, cc_extractor, cch, private_key
        )
        new_token, wm_logprob = basic_sample(wm_logits)
        # new_token: (batch_size, 1)
        output_logprobs.append(logprobs)
        watermarked_logprobs.append(wm_logprob)
        ccs.append(cc)
        watermark_codes.append(watermark_code)
        skippeds.append(skipped)
        input_tokens = new_token
        output_ids.append(new_token)
        input_ids = torch.cat([input_ids, new_token], dim=1)
        past_key_values = output.past_key_values
        if (new_token == model.config.eos_token_id).all():
            got_eos = True
            break
    output_ids = torch.cat(output_ids, dim=1)  # shape (batch_size, n)
    output_logprobs = torch.stack(
        output_logprobs, dim=1
    )  # shape (batch_size, n, vocab_size)
    watermarked_logprobs = torch.stack(watermarked_logprobs, dim=1)
    cc = np.stack(ccs, axis=-1)
    watermark_code = reweight.watermark_code_type.stack(watermark_codes, dim=-1)
    skipped = np.stack(skippeds, axis=-1)
    return (
        cc,
        watermark_code,
        skipped,
        output_ids,
        output_logprobs,
        watermarked_logprobs,
        past_key_values,
        got_eos,
    )


def basic_uwm_generator(
    reweight: AbstractReweight,
    cc_extractor: AbstractContextCodeExtractor,
    cch: ContextCodeHistory,
    private_key: bytes,
    model,
    input_ids: LongTensor,
    past_key_values=None,
    n=1,
    **kwargs
):
    model.eval()
    while True:
        (
            cc,
            watermark_code,
            skipped,
            output_ids,
            output_logprobs,
            watermark_logprobs,
            past_key_values,
            got_eos,
        ) = gen_n_token_uwm(
            reweight,
            cc_extractor,
            cch,
            private_key,
            model,
            input_ids,
            n,
            past_key_values=past_key_values,
            **kwargs,
        )
        yield output_ids, output_logprobs
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        if got_eos:
            break

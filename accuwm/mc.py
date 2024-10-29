from .utils import *
from .basic import gen_n_token


@torch.no_grad()
#  @profile_each_line
def gen_mc(
    model,
    input_ids: LongTensor,
    ref_output_ids: LongTensor,
    ref_logprobs: FloatTensor,
    past_key_values=None,
    process_logits_kwargs={},
) -> tuple[LongTensor, FloatTensor, FloatTensor, any, bool]:
    """
    model: Decoder-only model
    input_ids: (batch_size, seq_len). batch_size must be 1
    ref_output_ids: (batch_size, n)
    ref_logprobs: (batch_size, n, vocab_size)
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
    return: (output_ids, output_logprobs, poverlaps, past_key_values, got_eos)
    output_ids: (batch_size, gen_len)
    output_logprobs: (batch_size, gen_len, vocab_size)
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
    target_logits = logits[:, :-1, :]
    # target_logprob: (batch_size, n, vocab_size)

    gen_tokens, logprobs, poverlaps, fully_coupled = mc_sample(
        target_logits[0, :, :],  # shape (n, vocab_size)
        ref_logprobs[0],
        ref_output_ids[0],
    )
    # gen_tokens: (min(gen_len,n))
    # logprobs: (min(gen_len,n), vocab_size)
    # poverlaps: (min(gen_len,n))
    got_eos = False
    if gen_tokens[-1] == model.config.eos_token_id:
        got_eos = True
    if fully_coupled and not got_eos:
        new_token, logprobs_tail = basic_sample(logits[:, -1, :])
        # new_token: (batch_size=1, 1)
        output_ids = torch.cat([gen_tokens.unsqueeze(0), new_token], dim=-1)
        # output_ids: (1, n+1)
        output_logprobs = torch.cat(
            [logprobs.unsqueeze(0), logprobs_tail.unsqueeze(1)], dim=1
        )
        # output_logprobs: (1, n+1, vocab_size)
        if (new_token == model.config.eos_token_id).all():
            got_eos = True
    else:
        gen_len = gen_tokens.shape[0]
        output_ids = gen_tokens.unsqueeze(0)
        # output_ids: (1, gen_len)
        output_logprobs = logprobs.unsqueeze(0)
        # output_logprobs: (1, gen_len, vocab_size)
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
        output_ids,
        output_logprobs,
        poverlaps,
        past_key_values,
        got_eos,
    )


def mc_sample_generator_full(
    model,
    ref_model,
    input_ids: LongTensor,
    n: int,
    past_key_values=None,
    ref_past_key_values=None,
    **kwargs
):
    model.eval()
    ref_model.eval()
    while True:
        ref_output_ids, ref_logprobs, ref_past_key_values, _got_eos = gen_n_token(
            ref_model, input_ids, n, past_key_values=ref_past_key_values, **kwargs
        )
        #  output_ids, output_logprobs, poverlaps, past_key_values, got_eos = gen_mc_old(
        output_ids, output_logprobs, poverlaps, past_key_values, got_eos = gen_mc(
            model,
            input_ids,
            ref_output_ids,
            ref_logprobs,
            past_key_values=past_key_values,
            **kwargs,
        )
        ref_past_key_values = fix_gen_n_token_pass_key_values(
            ref_output_ids, output_ids, ref_past_key_values
        )
        yield output_ids, output_logprobs, poverlaps
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        if got_eos:
            break


def mc_sample_generator(*args, **kwargs):
    for output_ids, output_logprobs, poverlaps in mc_sample_generator_full(
        *args, **kwargs
    ):
        yield output_ids, output_logprobs

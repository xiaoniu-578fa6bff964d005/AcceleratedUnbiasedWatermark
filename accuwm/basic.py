from .utils import *


@torch.no_grad()
def gen_n_token(
    model,
    input_ids: LongTensor,
    n: int,
    past_key_values=None,
    process_logits_kwargs={},
) -> tuple[LongTensor, FloatTensor, any, bool]:
    """
    model: Decoder-only model
    input_ids: (batch_size, seq_len), need to be on the same device and appropriate dtype
    n: number of tokens to generate
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one or more token in input_ids
    return: (output_ids, output_logprobs, past_key_values, got_eos)
    output_ids: (batch_size, n)
    output_logprobs: (batch_size, n, vocab_size)
    past_key_values: following the format of huggingface's transformers. Doesn't cover last one token in output_ids
    got_eos: bool
    """
    if past_key_values is not None:
        # shape (batch_size, num_heads, n-1, head_dim)
        cached_n = past_key_values[0][0].shape[2]
        input_tokens = input_ids[:, cached_n:]
    else:
        input_tokens = input_ids
    output_ids = []
    output_logprobs = []
    device = model.device
    got_eos = False
    for i in range(n):
        output = model(
            input_tokens,
            past_key_values=past_key_values,
        )
        logits = output.logits[:, -1, :]
        logits = process_logits(input_ids, logits, **process_logits_kwargs)
        new_token, logprobs = basic_sample(logits)
        # new_token: (batch_size, 1)
        # logprobs: (batch_size, vocab_size)
        output_logprobs.append(logprobs)
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
    return output_ids, output_logprobs, past_key_values, got_eos


def basic_generator(
    model,
    input_ids: LongTensor,
    past_key_values=None,
    n=1,
    **kwargs,
):
    model.eval()
    while True:
        (
            output_ids,
            output_logprobs,
            past_key_values,
            got_eos,
        ) = gen_n_token(
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

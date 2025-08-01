# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/utils.py
# reference: https://github.com/lifeiteng/vall-e
from typing import Tuple

import torch
import torch.nn.functional as F


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    #>>> lengths = torch.tensor([1, 3, 2, 5])
    #>>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


def make_pad_mask_left(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    #>>> lengths = torch.tensor([1, 3, 2, 5])
    #>>> make_pad_mask(lengths)
    tensor(
        [
            [True,  True,  False],
            [True, False, False],
            [True,  True,  False],
            ...
        ]
    )
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).repeat(n, 1)
    expaned_lengths -= (max_len - lengths).unsqueeze(-1)

    return expaned_lengths < 0


# https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
    logits,
    top_k=0,
    top_p=1.0,
    filter_value=-float("Inf"),
    min_tokens_to_keep=1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep),
                    logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token


from typing import Optional


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[int] = None,
    repetition_penalty: float = 1.0,
):
    # if previous_tokens is not None:
    #     previous_tokens = previous_tokens.squeeze()
    # print(logits.shape,previous_tokens.shape)
    # pdb.set_trace()

    # 检查是否提供了 previous_tokens 和 repetition_penalty 是否不为 1.0。
    # torch.gather 用于从 logits 中根据 previous_tokens 提取对应的分数。
    # 根据分数的正负，应用不同的惩罚方式：负分数乘以 repetition_penalty，正分数除以 repetition_penalty。
    # logits.scatter_ 将惩罚后的分数放回 logits 中相应的位置。

    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=1, index=previous_tokens)
        score = torch.where(
            score < 0,
            score * repetition_penalty,
            score / repetition_penalty,
        )
        logits.scatter_(dim=1, index=previous_tokens, src=score)

    # 检查是否提供了 top_p 并且 top_p 是否小于 1.0。
    # torch.sort 对 logits 进行降序排序，得到排序后的 logits 和对应的索引 sorted_indices。
    # torch.nn.functional.softmax 将 sorted_logits 转换为概率分布，torch.cumsum 计算累积概率。
    # 根据累积概率与 top_p 的比较，生成需要移除的索引 sorted_indices_to_remove，并确保至少保留一个选项。
    # 将 sorted_indices_to_remove 散射到原始的索引位置 indices_to_remove。
    # 使用 logits.masked_fill 将需要移除的索引对应的 logits 值设为负无穷，这样在接下来的 softmax 操作中，这些位置的概率将接近 0。
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[:, 0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    # 调整 logits 的温度，温度越高，生成的词越随机。此处使用 max(temperature, 1e-5) 防止温度为 0 或负数导致的错误。
    # 操作确实不会改变元素的相对大小顺序，但会改变它们的绝对值大小，从而影响生成词的随机性。
    # 当 temperature 值较高时，logits 的值会被缩小，使得概率分布更加平滑。这意味着生成词的选择会更加随机，所有词的概率差异会被减小。
    # 当 temperature 值较低时，logits 的值会被放大，使得概率分布更加集中。这意味着生成词的选择会更加确定，高概率的词的概率会进一步提高，低概率的词的概率会进一步降低。
    logits = logits / max(temperature, 1e-5)

    # 检查是否提供了 top_k。
    # torch.topk 保留 logits 中概率最高的 top_k 个词。
    # pivot 是 logits 中第 top_k 高的分数。
    # 使用 torch.where 将低于 pivot 的分数设为负无穷，确保只有前 top_k 个词具有非零概率。
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v[:, -1].unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits, previous_tokens=previous_tokens, **sampling_kwargs)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    reference_free: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps -
                             reference_chosen_logps).detach()
    rejected_rewards = beta * \
        (policy_rejected_logps - reference_rejected_logps).detach()

    return losses.mean(), chosen_rewards, rejected_rewards


def get_batch_logps(
    logits_target: torch.FloatTensor,
    logits_reject: torch.FloatTensor,
    labels_target: torch.LongTensor,
    labels_reject: torch.LongTensor,
    average_log_prob: bool = False,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # dummy token; we'll ignore the losses on these tokens later

    per_token_logps_target = torch.gather(
        logits_target.log_softmax(-1), dim=2, index=labels_target.unsqueeze(2)
    ).squeeze(2)
    per_token_logps_reject = torch.gather(
        logits_reject.log_softmax(-1), dim=2, index=labels_reject.unsqueeze(2)
    ).squeeze(2)

    return per_token_logps_target.sum(-1), per_token_logps_reject.sum(-1)


def make_reject_y(y_o, y_lens):
    def repeat_P(y):
        range_idx, _ = torch.randint(0, len(y), size=(2,)).sort()
        pre = y[: range_idx[0]]
        shf = y[range_idx[1]:]
        range_text = y[range_idx[0]: range_idx[1]]
        new_y = torch.cat([pre, range_text, range_text, shf])
        return new_y

    def lost_P(y):
        range_idx, _ = torch.randint(0, len(y), size=(2,)).sort()
        pre = y[: range_idx[0]]
        shf = y[range_idx[1]:]
        range_text = y[range_idx[0]: range_idx[1]]
        new_y = torch.cat([pre, shf])
        return new_y

    bs = len(y_lens)
    reject_y = []
    reject_y_lens = []
    for b in range(bs):
        process_item_idx = torch.randint(0, 1, size=(1,))[0]
        if process_item_idx == 0:
            new_y = repeat_P(y_o[b])
            reject_y.append(new_y)
            reject_y_lens.append(len(new_y))
        elif process_item_idx == 1:
            new_y = lost_P(y_o[b])
            reject_y.append(new_y)
            reject_y_lens.append(len(new_y))
    max_length = max(reject_y_lens)
    for b in range(bs):
        pad_length = max_length - reject_y_lens[b]
        reject_y[b] = torch.cat([reject_y[b], torch.zeros(
            pad_length, dtype=y_o.dtype, device=y_o.device)], dim=0)

    reject_y = torch.stack(reject_y, dim=0)
    reject_y_lens = torch.tensor(reject_y_lens, device=y_lens.device)

    return reject_y, reject_y_lens

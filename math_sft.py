from typing import Union
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
import torch.nn as nn

'''
def sft():
    model = AutoModelForCausalLM.from_pretrained(
        "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        )
    data_loader = None
    loss_fn = torch.nn.functional.cross_entropy
    optimizer = torch.optim.AdamW
    gradient_accumulation_steps = 4
    for idx, (inputs, labels) in enumerate(data_loader):
        # Forward pass.
        logits = model(inputs)
        loss = loss_fn(logits, labels) / gradient_accumulation_steps
        # Backward pass.
        loss.backward()
        if (idx + 1) % gradient_accumulation_steps == 0:
        # Update weights every `gradient_accumulation_steps` batches.
        optimizer.step()
        # Zero gradients every `gradient_accumulation_steps` batches.
        optimizer.zero_grad()
'''



def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    '''
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for
    other tokens (prompt or padding).
    Args:
    prompt_strs: list[str] List of prompt strings.
    output_strs: list[str] List of output strings.
    tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.
    
    Returns:
    dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
    the tokenized prompt and output strings. Then the returned dictionary should have the
    following keys:
    
    input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
    the tokenized prompt and output strings, with the final token sliced off.
    
    labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
    shifted input ids, i.e., the input ids without the first token.
    response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -1): a mask on the response tokens in the labels.
    '''
    # Tokenize and pad
    output_encoded = tokenizer(
        output_strs,
        padding=True,  # Enable padding
        padding_side="right",
        truncation=False, # Enable truncation if sequences exceed max_length
        add_special_tokens=False,
        return_tensors="pt" # Return PyTorch tensors
    )

    input_encoded = tokenizer(
        prompt_strs,
        padding=True,  # Enable padding
        padding_side="left",
        truncation=False, # Enable truncation if sequences exceed max_length
        add_special_tokens=False,
        return_tensors="pt" # Return PyTorch tensors
    )

    prompt_ids, completion_ids = input_encoded["input_ids"], output_encoded["input_ids"]
    prompt_mask, completion_mask = input_encoded["attention_mask"], output_encoded["attention_mask"]
    input_ids = torch.cat((prompt_ids, completion_ids), dim=1)

    attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)
    completion_mask = torch.cat((torch.zeros_like(prompt_mask), completion_mask), dim=1)

    attention_mask, input_ids, completion_mask = flush_left(attention_mask, input_ids, completion_mask)
    
     # Create labels and mask padding tokens
    labels = input_ids.clone()
    #labels[attention_mask == 0] = -100
    #labels[completion_mask == 0] = -100
    response_mask = completion_mask.clone()
    tokenized_data = {
                        'input_ids': input_ids[:, :-1],
                        'labels': labels[:, 1:],
                        'response_mask': response_mask[:, 1:] > 0
                     }
    #print(tokenized_data)
    return tokenized_data

    
def flush_left(mask: torch.Tensor, *tensors: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
    """
    Shift non-zero elements in the mask and corresponding tensors to the left.

    This function operates on a binary mask and any number of additional tensors with the same dimensions as the mask.
    For each row, non-zero values are shifted to the leftmost positions. Then, columns that contain only zeros across
    all rows are truncated from the mask and tensors. Visually, this operation can be represented as follows:

    ```
    [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
     [0, x, x, x, 0, 0]]       [x, x, x, 0]]
    ```

    Args:
        mask (`torch.Tensor`):
            2D tensor (binary mask) with shape `(N, M)`.
        *tensors (`torch.Tensor`):
            One or more 2D tensors with the same shape as `mask`. These tensors will be processed alongside `mask`,
            with non-zero values shifted and excess zero columns truncated in the same manner.

    Returns:
        `torch.Tensor`:
            Updated binary mask with non-zero values flushed to the left and trailing zero columns removed.
        `*torch.Tensor`
            Updated tensors, processed in the same way as the mask.

    Example:
    ```python
    >>> mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    >>> tensor = torch.tensor([[9, 9, 2, 3, 4], [9, 5, 6, 9, 9]])
    >>> new_mask, new_tensor = flush_left(mask, tensor)
    >>> print(new_mask)
    tensor([[1, 1, 1],
            [1, 1, 0]])

    >>> print(new_tensor)
    tensor([[2, 3, 4],
            [5, 6, 0]])
    ```
    """
    _, M = mask.shape

    # Create copy of mask and tensors
    mask_copy = mask.clone()
    tensors = [t.clone() for t in tensors] # type: ignore

    # Shift non-zero values to the left
    first_non_zero = mask_copy.argmax(dim=1)
    pos = torch.arange(M, device=mask_copy.device).unsqueeze(0)
    idx_roll = (pos + first_non_zero.unsqueeze(1)) % M
    mask_roll = mask_copy.gather(1, idx_roll)
    rolled_tensors = [t.gather(1, idx_roll) for t in tensors]

    # Truncate trailing columns that are all zeros in mask_roll
    col_sums = mask_roll.sum(dim=0)
    empty_cols = col_sums == 0
    first_empty_col = int(empty_cols.to(torch.int8).argmax()) if empty_cols.any() else M
    flushed_mask = mask_roll[:, :first_empty_col]
    flushed_tensors = [t[:, :first_empty_col] for t in rolled_tensors]

    if not flushed_tensors:
        return flushed_mask
    return flushed_mask, *flushed_tensors

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    '''
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    Args:
    logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
    containing unnormalized logits.
    
    Returns:
    torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
    prediction.
    '''
    # Calculate per-token loss
    # In PyTorch's torch.nn.CrossEntropyLoss, setting reduce=False (or more commonly in newer versions, reduction='none') 
    # means that the loss is calculated for each individual sample in the batch, rather than being averaged or summed across the batch.
    # print(logits.shape)
    batch_size, seq_len, vocab_len = logits.shape # 2, 10, 100
    probabilities = nn.functional.softmax(logits, dim=-1) # 2, 10, 100
    log_prob = probabilities.clone()
    log_prob[..., :] = torch.log(probabilities[..., :])
    entropy = -(probabilities * log_prob) # calculation over last dim
    #print(entropy.shape) 
    return entropy.sum(-1) # 2, 10


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    '''
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
        and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
        response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
        tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling
        compute_entropy.

    Returns:
        dict[str, torch.Tensor].
        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
        log pθ(xt |x<t).
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
        for each position (present only if return_token_entropy=True)
    '''
    res = {}
    outputs = model(input_ids) # predicted logits
    outputs = outputs.logits
    batch_size, seq_len, vocab_len = outputs.shape
    probabilities = nn.functional.softmax(outputs, dim=-1)
    log_prob = probabilities.clone()
    log_prob[..., :] = torch.log(probabilities[..., :]) # 2, 10, 100

    # select the log prob of label
    '''
    # Indices to select from the last dimension for each element in the preceding dimensions
    # Shape: (batch_size, sequence_length)
    # For input_tensor[0, 0], select index 0 from the last dim (value 1)
    # For input_tensor[0, 1], select index 2 from the last dim (value 6)
    # And so on...
    indices_matrix = torch.tensor([
        [0, 2, 1],
        [1, 0, 2]
    ])

    # Create a range of indices for all dimensions except the last one,
    # then unsqueeze to align with the shape of indices_matrix for broadcasting
    # This creates a "mask" for the preceding dimensions
    '''
    batch_indices = torch.arange(batch_size).unsqueeze(1) # batch_indices will have shape (batch_size, 1)
    sequence_indices = torch.arange(seq_len).unsqueeze(0) # sequence_indices will have shape (1, sequence_length)
    res["log_probs"] = log_prob[batch_indices, sequence_indices, labels]
    if return_token_entropy:
        entropy = -(probabilities * log_prob) # calculation over last dim
        res["token_entropy"] = entropy.sum(-1) # 2, 10
    #print(log_prob.shape, res["log_probs"].shape)
    return res


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
    ) -> torch.Tensor:
    '''
    Sum over a dimension and normalize by a constant, considering only those elements where mask
    == 1.
    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all
        dimensions.

    Returns:
        torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to
        the sum.
    '''
    masked = tensor * mask
    masked = masked / normalize_constant
    if dim is not None:
        return masked.sum(dim=dim)
    else:
        return masked.sum()

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Execute a forward-and-backward pass on a microbatch.
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        SFT policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
        this so we can log it.
        metadata Dict with metadata from the underlying loss call, and any other statistics you
        might want to log
    '''
    # Forward pass, it is a mini batch
    loss = masked_normalize(policy_log_probs, response_mask, normalize_constant)
    # note for cross entropy -sum(p_true*log(p_pred)), p_true is as [0, 1, 0, ...], so log_probs per token is already the multiplcation
    loss = -1*loss / (gradient_accumulation_steps**2)
    loss.backward()
    #policy_log_probs.grad = policy_log_probs.grad * (gradient_accumulation_steps**2)
    meta_data = {}
    print("===> ", loss, policy_log_probs.grad)
    return (loss, meta_data)
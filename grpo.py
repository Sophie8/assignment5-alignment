from typing import Literal
import torch
import numpy as np
import wandb 
import copy

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from math_sft import log_generations, init_vllm, tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
    ):
    '''
    Compute rewards for each group of rollout responses, normalized by the group size.
    Args:
        reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against
        the ground truths, producing a dict with keys "reward", "format_reward", and
        "answer_reward".
        rollout_responses: list[str] Rollouts from the policy. The length of this list is
        rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str] The ground truths for the examples. The length of this
        list is rollout_batch_size, because the ground truth for each example is repeated
        group_size times.
        group_size: int Number of responses per question (group).
        advantage_eps: float Small constant to avoid division by zero in normalization.
        normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise
        subtract only the group mean.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
        advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
        response.
        raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
        metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards)
    '''
    raw_rewards = [] # rollout_batch_size,
    advantages_shapes = [] # rollout_batch_size,
    i = 0
    rewards_group = []
    #print(repeated_ground_truths)
    for rollout, truth in zip(rollout_responses, repeated_ground_truths):
        # for each rollout, we do group size times
        reward = reward_fn(rollout, truth)
        rewards_group.append(reward['reward'])
        if (i+1) % group_size == 0:
            # normalize within group
            data = np.array(rewards_group)
            if normalize_by_std:
                advantages_shape = (data - data.mean()) / (data.std() + advantage_eps)
                print("std ===> ", normalize_by_std, data, data.mean(), data.std(), advantage_eps)
            else:
                advantages_shape = data - data.mean()
            raw_rewards.append(copy.deepcopy(rewards_group))
            advantages_shapes.append(copy.deepcopy(advantages_shape))
            rewards_group = []
        i += 1
    raw_rewards = torch.tensor(raw_rewards, dtype=torch.float32)
    advantages_shapes = torch.tensor(advantages_shapes, dtype=torch.float32)
    # The dim=1 argument specifies that the mean should be calculated across the rows (the second dimension for a 2D tensor)
    metadata = {}
    # note squeeze only remove dim where dim size is 1, here we need to concatenate all lists to one
    '''
    When -1 is used as a dimension in view(), it tells PyTorch to automatically calculate the size of that 
    dimension based on the total number of elements
    # Flatten the tensor using .view(-1)
    flattened_tensor = original_tensor.view(-1)
    print("Flattened Tensor Shape:", flattened_tensor.shape)

    # Reshape into a 2D tensor with 2 rows, letting PyTorch infer columns
    reshaped_tensor = original_tensor.view(2, -1)
    print("Reshaped Tensor Shape:", reshaped_tensor.shape)
    '''
    return (advantages_shapes.view(-1), raw_rewards.view(-1), metadata)

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:
    '''
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
        reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
        each token.
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
        be aggregated across the batch and sequence dimensions in the training loop)
    '''
    # Broadcast the raw_rewards_or_advantages over the sequence_length dimension.
    # * operator do element wise multiplication with boardcasting, while @ is matrix multiplication: https://docs.pytorch.org/docs/stable/notes/broadcasting.html
    # https://medium.com/@krinaljoshi/broadcasting-in-pytorch-fc438ee04cc5, 
    # https://www.geeksforgeeks.org/deep-learning/understanding-broadcasting-in-pytorch/
    # rules: 1. Each tensor has at least one dimension.
    # 2. When iterating over the dimension sizes, starting at the trailing dimension, 
    # the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
    # If one tensor has a dimension of size 1, it can be "stretched" or "broadcast" across the corresponding dimension of the other tensor, effectively repeating its value along that dimension.
    # If one tensor has fewer dimensions than the other, the missing leading dimensions are implicitly treated as having a size of 1 for broadcasting purposes.
    '''
    example: data = tensor([[1., 2.],[3., 4.]]), scale = tensor([0.1000, 0.5000]), tmp = tensor([[0.1000],[0.5000]])
    as scale is (2,), tmp is (2, 1)
    if data * scale, then 2 matches 2, 1 is added to the leading position, scale becomes [0.1, 0.5],
                                                                                         [0.1, 0.5]
    if tmp * scale, then boardcasting happening at last dim 1, it becomes [0.1, 0.1],
                                                                           [0.5, 0.5]
    '''
    tmp = -1 * raw_rewards_or_advantages * policy_log_probs
    #print("==> ", raw_rewards_or_advantages.shape, policy_log_probs.shape, tmp.shape)
    return -raw_rewards_or_advantages * policy_log_probs # batch size * seq length



def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
        probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
        from the old policy.
        cliprange: float Clip parameter ε (e.g. 0.2).
    
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped
            loss.
            metadata dict containing whatever you want to log. We suggest logging whether each
            token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
            the min was lower than the LHS
    '''
    # note the difference between log prob and prob
    ratio = torch.exp(policy_log_probs - old_log_probs.detach()) # Detach ref_log_probs to stop gradients
    clipped_ratio= torch.clamp(ratio, min=1-cliprange, max=1+cliprange)
    surrogate_loss1 = ratio * advantages
    surrogate_loss2 = clipped_ratio * advantages
    loss = -torch.min(surrogate_loss1, surrogate_loss2)
    
    #print("===>: ", -1*ratio*advantages, -1*clipped_ratio*advantages)
    clipped_or_not = torch.eq(ratio, clipped_ratio)
    metadata = {"clipped_or_not":clipped_or_not}
   
    return (loss, metadata)


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Select and compute the desired policy-gradient loss.
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
        loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape
        (batch_size, 1).
        old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange Required for "grpo_clip"; scalar ε used for clipping.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss (batch_size, sequence_length), per-token loss.
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    '''
    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip"]
    metadata = {}
    if loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange) # type: ignore
    elif loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs) # type: ignore
    elif loss_type == "reinforce_with_baseline":
         # If  ̄r are the group-normalized rewards from compute_group_normalized_rewards (which
         # may or may not be normalized by the group standard deviation), then A =  -r
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs) # type: ignore
    return (loss, metadata)

def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
    ) -> torch.Tensor:
    '''
    Compute the mean of tensor along a given dimension, considering only those elements where
    mask == 1.
    Args:
        tensor: torch.Tensor The data to be averaged.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
        dim: int | None Dimension over which to average. If None, compute the mean over all
        masked elements.
    Returns:
        torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    '''
    masked = tensor * mask.float()
    num_unmasked_elements = mask.sum(dim=dim).float()
    masked_mean = masked.sum(dim = dim).float() / num_unmasked_elements
    return masked_mean

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    Execute a forward-and-backward pass on a microbatch.
    given the raw rewards or advantages and log probs, we will compute the per-token loss, use
    masked_mean to aggregate to a scalar loss per example, average over the batch dimension, adjust for gradient
    accumulation, and backpropagate
    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for
        prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step.
        loss_type One of "no_baseline", "reinforce_with_baseline", "grpo_clip".
        raw_rewards Needed when loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Needed when loss_type != "no_baseline"; shape (batch_size, 1).
        old_log_probs Required for GRPO-Clip; shape (batch_size, sequence_length).
        cliprange Clip parameter ε for GRPO-Clip.
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return
        this so we can log it.
        metadata Dict with metadata from the underlying loss call, and any other statistics you
        might want to log.
    '''
    # https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss_agg = masked_mean(loss, response_mask, dim=1)
    print("==> ", loss_agg.shape)
    # average across all batches
    loss_agg = loss_agg.mean()
    loss_agg = loss_agg / gradient_accumulation_steps
    loss_agg.backward()
    return loss_agg, metadata


def grpo_on_policy(experiment_name: str,
                   path_to_train_dataset: str, 
                   path_to_val_dataset: str,
                   model_path: str,
                   device='cuda'):
    '''
    Defaults
    '''
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4 # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1 # On-policy
    train_batch_size: int = 256 # On-policy
    gradient_accumulation_steps: int = 128 # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    #Initialize run
    run = wandb.init(
        entity=experiment_name,
        project="math qwen1.5b sft",
        group="SFT",  # all runs for the experiment in one group
    )
    log_generations()

    # load model and optimizer and inference
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        )
    vllm_eng = init_vllm(model_path, device, seed=42)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # load datasets
    train_data = load_dataset("parquet", data_files=path_to_train_dataset)
    val_data = load_dataset("parquet", data_files=path_to_val_dataset)
    print("total number of train data points: ", train_data)
    print("total number of validation data points: ", val_data)

    tokenizer = AutoTokenizer.from_pretrained("/home/shuyi/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2")
    train_data = train_data.map(lambda e: tokenize_prompt_and_output(e["prompt"], e["response"], tokenizer), batched=True)
    train_data.set_format(type='torch', columns=['input_ids', 'labels', 'response_mask'])
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=8)
    
    val_data = val_data.map(lambda e: tokenize_prompt_and_output(e["prompt"], e["response"], tokenizer), batched=True)
    val_data.set_format(type='torch', columns=['input_ids', 'labels', 'response_mask'])
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=8)

    # Training loop with gradient clipping
    num_epochs = 2
    clip_norm = 1.0  # The maximum allowed L2 norm of the gradients
    for epoch in range(num_epochs):
        for idx, (inputs, labels, response_masks) in enumerate(train_dataloader):
            # Forward pass.
            log_response = get_response_log_probs(model, inputs, labels, return_token_entropy=True)
            advantages, raw_rewards, _ = compute_group_normalized_rewards(reward_fn=r1_zero_reward_fn,
                                                                          log_response,
                                                                          repeated_ground_truth,
                                                                          group_size,
                                                                          advantage_eps,
                                                                          normalize_by_std=True)
            loss, meta_data = grpo_microbatch_train_step(log_response["log_probs"], 
                                                         response_masks,                     gradient_accumulation_steps, 
                                                         loss_type, 
                                                         advantages,
                                                         raw_rewards, 
                                                         old_log_probs, 
                                                         cliprange=None) # on policy no clipping
            # Backward pass.
            loss.backward()
            if (idx + 1) % gradient_accumulation_steps == 0:
                # Perform gradient clipping
                # This clips the L2 norm of the gradients of all model parameters
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                # Update weights every `gradient_accumulation_steps` batches.
                optimizer.step()
                # Zero gradients every `gradient_accumulation_steps` batches.
                optimizer.zero_grad()
            run.log({"train/prompt": inputs, "train/labels": labels})
            run.log(log_response)
from typing import Literal
import torch
import numpy as np

import copy


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
    # * operator do element wise boardcasting: https://docs.pytorch.org/docs/stable/notes/broadcasting.html
    return -1 * raw_rewards_or_advantages * policy_log_probs



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
from transformers import LogitsProcessor
import torch
import numpy as np


class RewardBasedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        num_candidates=5,
        reward_scale=0,
        reward_function=None,
        num_steps_to_apply_reward=0,
        tokenizer=None,
    ):
        self.num_candidates = num_candidates
        self.reward_scale = reward_scale
        self.reward_function = reward_function
        self.num_steps_to_apply_reward = num_steps_to_apply_reward
        self.tokenizer = tokenizer
        self.step_counter = 0

        assert self.num_candidates > 0
        assert reward_function is not None
        assert self.num_steps_to_apply_reward > 0
        assert tokenizer is not None

    def __call__(self, input_ids, scores):
        self.step_counter += 1

        if self.step_counter % self.num_steps_to_apply_reward == 0:
            # compute top-k candidates
            _, candidate_indices = scores.topk(self.num_candidates, dim=-1)

            # compute token_ids which are input_ids + candidate_index
            token_ids_list = []
            batch_size = input_ids.size(0)
            values = []
            for beam in range(batch_size):
                for candidate_index in candidate_indices[beam]:
                    token_ids = torch.cat([input_ids[beam], candidate_index.unsqueeze(0)])
                    token_ids_list.append(token_ids)
                    values.append(scores[beam, candidate_index].detach().item())
            candidate_texts = self.tokenizer.batch_decode(token_ids_list, skip_special_tokens=True)

            # compute rewards
            rewards = self.reward_function(candidate_texts)
            rewards = torch.tensor(rewards, dtype=scores.dtype, device=scores.device)
            rewards = rewards.reshape(batch_size, self.num_candidates, -1)

            # compute total scores
            reward_guided_scores = torch.ones_like(scores) * float("-inf")
            for beam in range(batch_size):
                for index, reward in zip(candidate_indices[beam], rewards[beam]):
                    reward_guided_scores[beam, index] = scores[beam, index] + self.reward_scale * reward
            scores = reward_guided_scores

        return scores


class FrequencyBasedLogitsProcessor(LogitsProcessor):
    def __init__(self, frequencies, alpha=0.7, beta=0.5):
        self.normalized_frequencies = beta * torch.nn.functional.softmax(alpha * frequencies, dim=-1)

    def __call__(self, input_ids, scores):
        return scores * (1.0 + self.normalized_frequencies)

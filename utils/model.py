from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BeamSearchScorer,
    LogitsProcessorList,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
import torch

from utils.logits_processor import (
    RewardBasedLogitsProcessor,
    FrequencyBasedLogitsProcessor,
)


MAX_NEW_TOKENS = 16


class Model:
    def __init__(self, model_name_or_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def decode_greedy(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=self.tokenizer.eos_token_id,  # Mistral
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs

    def decode_beam(self, prompt, num_beams=4):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            early_stopping=True,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=self.tokenizer.eos_token_id,  # Mistral
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs

    def decode_contrastive(self, prompt, num_beams=4, penalty_alpha=0.6, top_k=4):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            temperature=0.7,
            do_sample=True,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            early_stopping=True,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=self.tokenizer.eos_token_id,  # Mistral
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs

    def decode_nucleus(self, prompt, temperature=0.7, top_p=0.9):
        inputs = self.tokenizer([prompt] * 4, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            sampling=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs

    def decode_reward_guided(
        self,
        prompt,
        num_beams=4,
        num_candidates=10,
        reward_scale=0.0,
        reward_function=None,
        num_steps_to_apply_reward=2,
    ):
        inputs = self.tokenizer([prompt] * num_beams, return_tensors="pt").to(self.model.device)

        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=num_beams,
            num_beam_hyps_to_keep=num_beams,
            device=self.model.device,
        )

        logits_processor = LogitsProcessorList()
        logits_processor.append(
            RewardBasedLogitsProcessor(
                num_candidates=num_candidates,
                reward_scale=reward_scale,
                reward_function=reward_function,
                num_steps_to_apply_reward=num_steps_to_apply_reward,
                tokenizer=self.tokenizer,
            )
        )

        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=inputs["input_ids"].shape[-1] + MAX_NEW_TOKENS))

        outputs = self.model.beam_search(
            **inputs,
            beam_scorer=beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,  # Mistral
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs

    def decode_frequency_guided(self, prompt, num_beams=4, frequencies=None, alpha=0.7, beta=0.7):
        beam_scorer = BeamSearchScorer(
            batch_size=1,
            num_beams=num_beams,
            num_beam_hyps_to_keep=num_beams,
            device=self.model.device,
        )

        logits_processor = LogitsProcessorList()
        logits_processor.append(
            FrequencyBasedLogitsProcessor(
                frequencies=frequencies,
                alpha=alpha,
                beta=beta,
            )
        )
        inputs = self.tokenizer([prompt] * num_beams, return_tensors="pt").to(self.model.device)

        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=inputs["input_ids"].shape[-1] + MAX_NEW_TOKENS))

        outputs = self.model.beam_search(
            **inputs,
            beam_scorer=beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return outputs

import argparse
from datasets import load_dataset
import json
import os
from tqdm import tqdm
import torch

from utils.model import Model
from utils.constants import (
    QA_TEMPLATE_WITH_CONTEXT,
    QA_TEMPLATE_WITH_CONTEXT_PREFIX,
    QA_TEMPLATE_WITHOUT_CONTEXT,
    QA_TEMPLATE_WITHOUT_CONTEXT_PREFIX,
)
from utils.reward_functions import SimilarityRewardMax, SimilarityRewardMin


args = argparse.ArgumentParser()
args.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
args.add_argument(
    "--method",
    type=str,
    default="greedy",
    choices=["greedy", "beam", "contrastive", "nucleus", "reward", "frequency"],
)
args.add_argument("--num_beams", type=int, default=4)
args.add_argument("--num_candidates", type=int, default=10)
args.add_argument("--reward_scale", type=float, default=1.5)
args.add_argument("--alpha", type=float, default=0.7)
args.add_argument("--beta", type=float, default=0.7)
args.add_argument("--num_contexts", type=int, default=3)
args.add_argument("--without_contexts", action="store_true", default=False)
args = args.parse_args()

# load dataset
dataset = json.load(open("trivia_qa/rc_wikipedia_validation.json"))

# load model
model = Model(args.model_name_or_path)

# get responses
os.makedirs("trivia_qa", exist_ok=True)
results = []
for id, sample in tqdm(dataset.items()):

    if id == "metadata":
        continue

    # if int(id) > 1:
    #     break

    question = sample["question"]
    contexts = sample["contexts"][: args.num_contexts]
    answers = sample["answers"]

    if args.without_contexts:
        prompt = QA_TEMPLATE_WITHOUT_CONTEXT.format(
            question=question,
        )
        prompt_prefix = QA_TEMPLATE_WITHOUT_CONTEXT_PREFIX.format(
            question=question,
        )
        run_name = f"{args.model_name_or_path.replace('/', '_')}_{args.method}_without_contexts"
    else:
        prompt = QA_TEMPLATE_WITH_CONTEXT.format(
            question=question,
            contexts="\n\n".join(contexts),
        )
        prompt_prefix = QA_TEMPLATE_WITH_CONTEXT_PREFIX.format(
            question=question,
            contexts="\n\n".join(contexts),
        )
        run_name = f"{args.model_name_or_path.replace('/', '_')}_{args.method}"

    if args.method == "greedy":
        responses = model.decode_greedy(prompt)
    elif args.method == "beam":
        responses = model.decode_beam(prompt)
    elif args.method == "contrastive":
        responses = model.decode_contrastive(prompt)
    elif args.method == "nucleus":
        responses = model.decode_nucleus(prompt)
    elif args.method == "reward":
        run_name = run_name + f"_{args.num_beams}_{args.num_candidates}_{args.reward_scale}"
        reward_function = SimilarityRewardMax(
            contexts,
            prefix=prompt_prefix,
        )
        responses = model.decode_reward_guided(
            prompt,
            num_beams=args.num_beams,
            num_candidates=args.num_candidates,
            reward_scale=args.reward_scale,
            reward_function=reward_function,
            num_steps_to_apply_reward=2,
        )

    results.append(
        {
            "id": id,
            "question": question,
            "contexts": contexts,
            "answers": answers,
            "prompt": prompt,
            "responses": responses,
        }
    )
    json.dump(results, open(f"trivia_qa/{run_name}.json", "w"), indent=4)
    # print("\n\n".join([response[-200:] for response in responses]))

import json
from tqdm import tqdm


def show_runs(runs):
    for run in runs:
        outputs = json.load(open("nq/{}.json".format(run)))

        results = []
        for id, sample in enumerate(outputs):
            prompt = sample["prompt"]
            answers = sample["answers"]
            responses = sample["responses"]

            # if int(id) > 50:
            #     continue

            answers = ["".join([c if c.isalnum() else " " for c in answer]).lower() for answer in answers]

            good = False
            for response in responses:
                response = response.removeprefix(prompt)
                response = "".join([c if c.isalnum() else " " for c in response]).lower()
                for answer in answers:
                    good = good or (answer in response)
            results.append(good)

        print("{:45s} {:.2f}".format(run, sum(results) / len(results) * 100))


show_runs(
    [
        "huggyllama_llama-7b_greedy",
        "huggyllama_llama-7b_beam",
        "huggyllama_llama-7b_contrastive",
        # "huggyllama_llama-7b_nucleus",
        # "huggyllama_llama-7b_reward_4_10_0.1",
        "huggyllama_llama-7b_reward_4_10_0.5",
        # "huggyllama_llama-7b_reward_4_10_1.0",
        "huggyllama_llama-7b_reward_4_10_2.0",
        # "huggyllama_llama-7b_reward_4_10_3.0",
        # "huggyllama_llama-7b_reward_4_10_4.0",
        "huggyllama_llama-7b_reward_4_10_5.0",
        "huggyllama_llama-7b_reward_4_10_10.0",
    ]
)
print()
show_runs(
    [
        "google_gemma-7b_greedy",
        "google_gemma-7b_beam",
        "google_gemma-7b_contrastive",
        # "google_gemma-7b_nucleus",
        "google_gemma-7b_reward_4_10_0.5",
        "google_gemma-7b_reward_4_10_2.0",
        "google_gemma-7b_reward_4_10_5.0",
        "google_gemma-7b_reward_4_10_10.0",
    ]
)
print()
show_runs(
    [
        "meta-llama_Llama-2-7b-hf_greedy",
        "meta-llama_Llama-2-7b-hf_beam",
        "meta-llama_Llama-2-7b-hf_contrastive",
        # "meta-llama_Llama-2-7b-hf_nucleus",
        "meta-llama_Llama-2-7b-hf_reward_4_10_0.5",
        "meta-llama_Llama-2-7b-hf_reward_4_10_2.0",
        "meta-llama_Llama-2-7b-hf_reward_4_10_5.0",
        "meta-llama_Llama-2-7b-hf_reward_4_10_10.0",
    ]
)
# print()
# show_runs(
#     [
#         "mistralai_Mistral-7B-v0.1_greedy",
#         "mistralai_Mistral-7B-v0.1_beam",
#         "mistralai_Mistral-7B-v0.1_contrastive",
#         # "meta-llama_Llama-2-7b-hf_nucleus",
#         "mistralai_Mistral-7B-v0.1_reward_4_10_0.5",
#         "mistralai_Mistral-7B-v0.1_reward_4_10_2.0",
#         "mistralai_Mistral-7B-v0.1_reward_4_10_5.0",
#         "mistralai_Mistral-7B-v0.1_reward_4_10_10.0",
#     ]
# )

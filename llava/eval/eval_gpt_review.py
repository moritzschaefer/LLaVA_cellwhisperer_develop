import argparse
import json
import os
import numpy as np


import openai
import tqdm
import time

client = openai.OpenAI(api_key=os.getenv("API_KEY"))  # Initialize the OpenAI client


def get_eval(system_prompt, content: str, max_tokens: int, model: str):
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)

    # return response['choices'][0]['message']['content']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument("--model")
    # parser.add_argument("--api-base-url")
    parser.add_argument("-q", "--question")
    # parser.add_argument('-a', '--answer')
    parser.add_argument("-a", "--answer-list", nargs="+", default=[])
    parser.add_argument("-r", "--rule")
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    args = parser.parse_args()

    f_q = open(os.path.expanduser(args.question))
    f_ans = [open(os.path.expanduser(ans_f)) for ans_f in args.answer_list]
    answer_names = [os.path.basename(ans_f).split(".")[0] for ans_f in args.answer_list]
    rule_dict = json.load(open(os.path.expanduser(args.rule), "r"))

    review_file = open(f"{args.output}", "w")

    for idx, ques_ans in enumerate(tqdm.tqdm(zip(f_q, *f_ans))):
        ques = json.loads(ques_ans[0])
        ans = [json.loads(ans_js) for ans_js in ques_ans[1:]]

        # shuffle the answers to avoid bias. store the shuffling indices to restore later
        idx_shuffle = np.random.permutation(len(ans))
        revert_idx_shuffle = np.argsort(idx_shuffle)
        shuffled_ans = [ans[i] for i in idx_shuffle]

        category = json.loads(ques_ans[0]).get("category")
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            rule = rule_dict["default"]
        system_prompt = rule["prompt"]
        role = rule["role"]
        content = (
            f'[Question]\n{ques["text"]}\n\n[End of Question]\n\n'
            f'[Reference]\n{ques["reference"]}\n\n[End of Reference]\n\n'
        )

        for i, ans_i in enumerate(shuffled_ans):
            content += f'[{role} {i+1}]\n{ans_i["text"]}\n\n[End of {role} {i+1}]\n\n'

        res = get_eval(system_prompt, content, args.max_tokens, model=args.model)

        review_dict = json.loads(res)

        # store the reverted order for the answers
        write_dict = {
            "id": idx + 1,  # why not start with 0?
            "question_id": ques["question_id"],
            "answer_ids": [ans_i["answer_id"] for ans_i in ans],
            "content": review_dict["explanation"],
            "scores": {
                answer_name: review_dict[f"assistant_{reverted_score+1}"]
                for answer_name, reverted_score in zip(answer_names, revert_idx_shuffle)
            },
            "category": category,
        }
        review_file.write(json.dumps(write_dict) + "\n")

    review_file.close()

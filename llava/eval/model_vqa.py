"""
Note. Currently no support for `mm_vision_select_layer`
"""
import argparse
import torch
import os
import json
import numpy as np

from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device="cuda")

    # Questions
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)  # no-op with default args


    # Load image data
    image_data = dict(np.load(args.image_data, allow_pickle=True))

    nans = np.where(np.isnan(image_data["transcriptome_embeds"]))
    broken_ids = image_data["orig_ids"][np.unique(nans[0])]
    assert len(broken_ids) == 0, f"Found broken ids: {broken_ids}"

    orig_id_to_int = {k: v for k, v in zip(image_data["orig_ids"], range(len(image_data["orig_ids"])))}
    assert all([q["image"] in orig_id_to_int for q in questions])

    # Prepare answers file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_id = line["image"]
        qs = line["text"]
        if isinstance(qs, str):
            qs = [qs]

        cur_prompt = qs[-1]  # not sure if this is what we want

        if model.config.mm_use_im_start_end:
            qs[0] = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs[0]
        else:
            qs[0] = DEFAULT_IMAGE_TOKEN + '\n' + qs[0]

        conv = conv_templates[args.conv_mode].copy()
        assert len(qs) % 2 == 1, "Must end with a user request"
        for i, q in enumerate(qs):
            conv.append_message(conv.roles[i%2], q)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        image_tensor = torch.tensor(image_data["transcriptome_embeds"][orig_id_to_int[image_id]], device=model.device, dtype=torch.bfloat16).unsqueeze(0)  # We are using bfloat16

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id,  # explicitly request open-ended generation (suppresses warnings)
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-data", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)


"""
Preprocess the DeepMath dataset to parquet format
"""

import os
import json
import datasets
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.utils.reward_score.math_verify import compute_score
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/DeepMath-Qwen-base')
    parser.add_argument('--model_type', default='base')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    data_source = 'DeepMath-Qwen-base'
    print(f"Loading the {data_source} dataset from HuggingFace...", flush=True)
    ds = datasets.load_dataset("zwhe99/DeepMath-103K")

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('question') # There are instructions in the prompt.
            answer = example.pop('final_answer')
            question = question_raw + " " + instruction_following
            if args.model_type == 'base':
                prompt = question
            elif args.model_type == 'instruct':
                prompt = [{
                    "content":question,
                    "role": "user"
                }]
            else:
                raise ValueError(f"Invalid model type: {args.model_type}")

            return {
                "data_source":  data_source,
                "prompt":       prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'index': 'DeepMath-' + str(idx)
                }
            }
        return process_fn

    model_type = args.model_type

    train_dataset = ds['train'].map(function=make_map_fn('train'), with_indices=True)
    import pdb; pdb.set_trace()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)


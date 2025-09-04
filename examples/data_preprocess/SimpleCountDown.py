# create the count down dataset

import os
import json
import datasets
from datasets import Dataset
import random
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
from collections import defaultdict


def generate_simple_count_down_dataset(num_samples, max_num, difficulty):
    """
    Generate a simple count down dataset
    num_samples: number of distinct questions.
    max_num: the maximum number to count down from. The absolute value of the given integers in the sequence is no larger than the max_num.
    difficulty: the difficulty level of the dataset. This is the length of the input sequence.
    """
    dataset = []
    seen = set()

    for i in tqdm(range(num_samples), desc=f"Generating simple count down dataset with difficulty = {difficulty}"):
        sequence, operator_sequence, answer = generate_single_question_dataset(max_num, difficulty)
        if (tuple(sequence), tuple(operator_sequence)) in seen:
            continue
        seen.add((tuple(sequence), tuple(operator_sequence)))

        question = f"Using the numbers {sequence}, create an equation that equals {answer}. You can use plus and minus (+, -) and each number must be used once and can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> 1 + 2 - 3 </answer>. Let's think step by step and output the final answer."

        dataset.append(
            {
                "data_source": "SimpleCountDown",
                # "prompt": [{
                #     "content": question,
                #     "role": "user"
                # }],
                "prompt": question,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": answer,
                        "numbers": sequence
                    }
                },
                "extra_info": {"index": i, "level": difficulty}
            }
        )
    return dataset

def generate_single_question_dataset(max_num, difficulty):
    """
    Generate a single question dataset
    max_num: the maximum number to count down from. The absolute value of the given integers in the sequence is no larger than the max_num.
    difficulty: the difficulty level of the dataset. This is the length of the input sequence.
    """
    sequence = []
    for i in range(difficulty):
        sequence.append(random.randint(1, max_num))
    operator_sequence = []
    for i in range(difficulty - 1):
        operator_sequence.append(random.choice(['+', '-']))
    answer = sequence[0]
    for i in range(difficulty - 1):
        if operator_sequence[i] == '+':
            answer += sequence[i + 1]
        else:
            answer -= sequence[i + 1]
    return sequence, operator_sequence, answer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/simple_count_down_base')
    parser.add_argument('--model_type', default='base')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    min_difficulty = 2
    max_difficulty = 9
    max_num = 1000
    num_samples = 102000

    data_dict = defaultdict(list)
    for difficulty in range(min_difficulty, max_difficulty + 1):
        dataset = generate_simple_count_down_dataset(num_samples, max_num, difficulty)

        train_dataset = dataset[:-2000]
        test_dataset = dataset[-2000:]

        data_dict[f'count_down_L{difficulty}_train'] = train_dataset
        data_dict[f'count_down_L{difficulty}_test'] = test_dataset

        train_dataset = Dataset.from_list(train_dataset)
        train_dataset.to_parquet(os.path.join(args.local_dir, f'count_down_L{difficulty}_train.parquet'))
        if args.hdfs_dir is not None:
            makedirs(args.hdfs_dir)
            copy(os.path.join(args.local_dir, f'count_down_L{difficulty}_train.parquet'), args.hdfs_dir)

        test_dataset = Dataset.from_list(test_dataset)
        test_dataset.to_parquet(os.path.join(args.local_dir, f'count_down_L{difficulty}_test.parquet'))
        if args.hdfs_dir is not None:
            makedirs(args.hdfs_dir)
            copy(os.path.join(args.local_dir, f'count_down_L{difficulty}_test.parquet'), args.hdfs_dir)

    # mix the datasets
    difficulties = list(range(min_difficulty, max_difficulty + 1))  # [2,3,…,9]
    # raw mixture weights for each difficulty
    mixture_weights = {
        'easy':   [max_difficulty - d + 1 for d in difficulties],  # [8,7,…,1]
        'middle': [1 for _ in difficulties],                       # [1,1,…,1]
        'hard':   [d - min_difficulty + 1 for d in difficulties],  # [1,2,…,8]
    }
    random.seed(42)

    for mix_name, raw_w in mixture_weights.items():
        total_w = sum(raw_w)
        mixed_train, mixed_test = [], []

        for d, w in zip(difficulties, raw_w):
            # fetch the lists you stored earlier
            train_list = data_dict[f'count_down_L{d}_train']
            test_list  = data_dict[f'count_down_L{d}_test']

            # decide how many to draw
            train_count = int(round(w / total_w * 400000))
            test_count  = int(round(w / total_w * 2000))

            sel_train = random.sample(train_list, train_count)
            sel_test = random.sample(test_list, test_count)
            mixed_train.extend(sel_train)
            mixed_test.extend(sel_test)

        # shuffle the final mixes
        random.shuffle(mixed_train)
        random.shuffle(mixed_test)

        # save to parquet
        mixed_train_ds = Dataset.from_list(mixed_train)
        mixed_test_ds  = Dataset.from_list(mixed_test)

        train_path = os.path.join(args.local_dir, f'count_down_mix_{mix_name}_train.parquet')
        test_path  = os.path.join(args.local_dir, f'count_down_mix_{mix_name}_test.parquet')

        mixed_train_ds.to_parquet(train_path)
        mixed_test_ds.to_parquet(test_path)

        # copy to HDFS if requested
        if args.hdfs_dir:
            makedirs(args.hdfs_dir)
            copy(train_path, args.hdfs_dir)
            copy(test_path,  args.hdfs_dir)

        print(f"→ saved mixture `{mix_name}`: train={len(mixed_train)} / test={len(mixed_test)}")

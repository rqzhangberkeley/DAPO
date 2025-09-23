import os
import argparse
import pandas as pd
import datasets
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/AIME2025-dup16-Qwen-base')
    parser.add_argument('--model_type', default='base')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_copies', type=int, default=16)
    args = parser.parse_args()

    # Read the original dataset
    dataset_path = 'rqzhang/AIME2025-Qwen-base'
    dataset = datasets.load_dataset(dataset_path)
    split = 'train'

    # Make copies and combine
    datasets_list = [dataset[split] for _ in range(args.num_copies)]
    combined_dataset = datasets.concatenate_datasets(datasets_list)
    shuffled_dataset = combined_dataset.shuffle()

    # Create output directory if it doesn't exist
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Save files
    output_parquet = os.path.join(args.local_dir, f'{split}.parquet')
    output_jsonl = os.path.join(args.local_dir, f'{split}.jsonl')
    
    shuffled_dataset.to_parquet(output_parquet)
    shuffled_dataset.to_json(output_jsonl, orient="records", lines=True)

    print(f"Created {args.num_copies}x copies of {dataset_path}, shuffled and saved to {args.local_dir}")

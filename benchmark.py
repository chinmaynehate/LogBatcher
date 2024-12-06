import argparse
import json
import os
import pandas as pd
from logbatcher.parser import Parser
from logbatcher.util import generate_logformat_regex, log_to_dataframe
from logbatcher.parsing_base import single_dataset_paring

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='2k', choices=['2k', 'full'],
                        help='Evaluate on 2k or full dataset.')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='The Large Language models to benchmark, e.g., gpt-3.5-turbo-0125 gpt-4o-mini-2024-07-18.')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='The size of a batch.')
    parser.add_argument('--sample_method', type=str, default='dpp', choices=['dpp', 'random', 'similar'],
                        help='Sample method: dpp, random, similar.')
    parser.add_argument('--chunk_size', type=int, default=2000,
                        help='Size of logs in a chunk.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='The dataset to evaluate on.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_args()

    # Define datasets
    datasets = ['BGL', 'HDFS', 'OpenStack', 'OpenSSH', 'HPC', 'Zookeeper', 'Spark', 
                'Proxifier', 'HealthApp', 'Mac', 'Hadoop', 'Apache', 'Linux', 
                'Thunderbird', 'Windows', 'Android']

    if args.dataset not in datasets:
        print(f"Dataset {args.dataset} not recognized. Please choose from: {datasets}")
        exit(1)

    # Output directory setup
    theme = f"logbatcher_{args.data_type}"
    output_dir = f'outputs/parser/{theme}/'
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)

    for model in args.models:
        print(f"\nBenchmarking with model: {model}")
        parser = Parser(model, theme, config)

        dataset = args.dataset
        print(f"Processing dataset: {dataset} with model: {model}")

        # Load structured logs
        if args.data_type == '2k':
            structured_log_file = f'datasets/loghub-2k/{dataset}/{dataset}_2k.log_structured_corrected.csv'
        elif args.data_type == 'full':
            structured_log_file = f'datasets/loghub-2.0/{dataset}/{dataset}_full.log_structured.csv'
        else:
            raise ValueError('data_type should be 2k or full')

        if not os.path.exists(structured_log_file):
            print(f"Log file {structured_log_file} not found.")
            continue

        df = pd.read_csv(structured_log_file)
        logs = df['Content'].tolist()

        single_dataset_paring(
            dataset=dataset,
            contents=logs,
            output_dir=output_dir, 
            parser=parser, 
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            sample_method=args.sample_method,
            data_type=args.data_type
        )

        print(f"Parsing completed for model: {model}")
        print(f"Time cost by LLM ({model}): {parser.time_consumption_llm} seconds")

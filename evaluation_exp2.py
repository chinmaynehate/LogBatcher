import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import pandas as pd
from tqdm import tqdm
from utils.cluster import Cluster,tokenize, vectorize, cluster, reassign_clusters
from utils.parser import Cluster_Parser
from utils.evaluator import evaluate, evaluate_all_datasets
from utils.sample import sample_from_clusters
from tqdm import tqdm




def single_dataset_paring(dataset, output_dir, parser, shot, candidate, batch_size, chunk_size ,Concurrent=True, sample_method = 'dpp'):
    print(f'Parsing {dataset}...')

    # initialize
    df = pd.read_csv(f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv')
    logs = df['Content'].tolist()
    logs_list = [logs[i:i+chunk_size] for i in range(0, len(logs), chunk_size)]
    outputs = [None for _ in range(len(logs))]
    tmps_list = [None for _ in range(len(logs))]
    cache_pairs = []
    for chunk_index,log_chunk in enumerate(logs_list):

        print(f"-" * 40)
        print(f"parsing the chunk {chunk_index}")
        print(f"-" * 40)

        # tokenize -> vectorize -> cluster -> reassign_clusters
        tokenized_logs = [tokenize(log) for log in log_chunk]
        labels, cluster_nums = cluster(vectorize(tokenized_logs))
        labels, cluster_nums = reassign_clusters(labels, cluster_nums, tokenized_logs)
        
        # store the logs in each cluster and sort them by the number of logs in each cluster
        inputs = []
        clusters = []
        for i in range(cluster_nums):
            inputs.append([-1, [], [], '']) # label, logs, indexs, oracle_template
        for i, label in enumerate(labels):
            i = i + chunk_index * chunk_size
            inputs[label][0] = label
            inputs[label][1].append(logs[i])
            inputs[label][2].append(i)
            inputs[label][3] = ''
        for input in inputs:
            c = Cluster(*input, remove_duplicate=True,remain_num=batch_size, sample_method=sample_method)
            clusters.append(c)
        clusters = sorted(clusters, key=lambda cluster: len(cluster.indexs), reverse=True)
    

        # parse each cluster
        for index, c in enumerate(tqdm(clusters)):
            # print(f"=" * 40)
            # print(f"parsing the cluster {index} in {cluster_nums} clusters\nsample log: {c.logs[0]}")
            tmp, template, c, new_cluster = parser.get_responce( c, cluster_nums, cache_pairs, [], shot)

            # update clusters
            if new_cluster != None:
                clusters.append(new_cluster)
                cluster_nums += 1

            # update cache
            template_exist = any(pair[1] == template for pair in cache_pairs)
            if not template_exist and template != '<*>' and template.strip() != '':
                cache_pairs.append([c.logs[0],template])

            for index in c.indexs:
                outputs[index] = template
                tmps_list[index] = tmp
    t2 = time.time()
    # write to file
    df_new = pd.DataFrame()
    df_new['Content'] = logs
    df_new['EventTemplate'] = outputs
    df_new.to_csv(output_dir + f'{dataset}_2k.log_structured.csv', index=False)
    evaluate(output_dir + f'{dataset}_2k.log_structured.csv',f'dataset/{dataset}/{dataset}_2k.log_structured_corrected.csv', dataset)

def set_args():
    # 定义命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125',
                        help='use which model to parse the log.')
    parser.add_argument('--candidate', type=int, default=32,
                        help='The num of candidate pairs.')
    parser.add_argument('--shot', type=int, default=0,
                        help='The num of demostrations.')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='The size of a batch')
    parser.add_argument('--sample_method', type=str, default='dpp',
                        help='Sample method: dpp, random, similar.')
    parser.add_argument('--chunk_size', type=int, default=200,
                        help='Size of logs in a chunk')
    # 解析命令行参数
    args = parser.parse_args()
    # 调用处理函数
    return args


if __name__ == "__main__":
    args = set_args()
    # datasets = ['BGL', 'HDFS', 'HealthApp', 'OpenStack', 'OpenSSH', 'HPC', 'Zookeeper',
    #             'Mac', 'Hadoop', 'Android', 'Windows', 'Apache', 'Thunderbird', 'Spark', 'Linux']
    datasets = ['HDFS']
    model = args.model
    
    theme = f"LogBatcher_{args.shot}shot_{args.candidate}candidate_{args.batch_size}batchsize_{args.chunk_size}chunksize_full"

    output_dir = f'outputs/parser/{theme}/'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # else:
    #     print(f'{output_dir} already exists.\nresults is here: {output_dir}')
    #     exit()
    with open('config.json', 'r') as f:
        config = json.load(f)
    config['model'] = args.model
    
    for index, dataset in enumerate(datasets):
        parser = Cluster_Parser(theme, config)
        single_dataset_paring(
            dataset=dataset, 
            output_dir=output_dir, 
            parser=parser, 
            shot=args.shot,
            candidate=args.candidate,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            Concurrent=False,
            sample_method = args.sample_method
            
        )
        print('time cost by llm: ', parser.time_consumption)
    evaluate_all_datasets(theme)
import pandas as pd
from utils import load_data
import numpy as np
import torch
import copy
import glob
from scipy.spatial.distance import cosine, euclidean, cityblock
import pickle as pkl
import igraph
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import cosine, euclidean, correlation, cityblock
similarity_list = [cosine, euclidean, correlation, cityblock]

import argparse


if __name__ == "__main__":
    # load saved results, get poisoning edges
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="GCN", help='Model Type, GAT or gcn')
    parser.add_argument('--dataset', type=str, default="citeseer", help='dataset: citeseer, pokec and cora')
    parser.add_argument('--num_target', type=float, default=0.01, help='int or percentage')
    parser.add_argument('--beta', type=float, default=1, help='1-5')

    args = parser.parse_args()
    model = args.model  # "gcn"
    dataset = args.dataset
    result_suffix = f'_num={args.num_target}'
    # load data
    adj_ori, ft, labels = load_data("data", dataset, dropout=0)


    with open(f"{model}/{dataset}/add_edge_experiments{result_suffix}_exp2.pkl", 'rb') as f:
        prepared_results = pkl.load(f)
    add_noise_to_nodes = set()
    for e in prepared_results['added_edges']:
        add_noise_to_nodes.add(e[0])
        add_noise_to_nodes.add(e[1])

    add_noise_to_nodes = list(add_noise_to_nodes)
    # load target model output: PU
    target_output_df = pd.read_csv(f"{model}/{dataset}/target-output-0.csv")
    # load target model output: Pp
    target_output_df_p = pd.read_csv(f"{model}/{dataset}/target-output-{0}-{'exp2' + result_suffix}.csv")

    # add noise to target_df

    target_output_df.loc[add_noise_to_nodes] = target_output_df.loc[add_noise_to_nodes] + np.random.normal(0, beta, target_output_df.loc[add_noise_to_nodes].shape)

    # normalize: sum should be 1
    target_output_df = target_output_df.div(target_output_df.sum(axis=1), axis=0)

    #construct attack feature

    targets_exp2 = prepared_results['targets']

    PPu = target_output_df_p.loc[[x[0] for x in targets_exp2]].values
    PPv = target_output_df_p.loc[[x[1] for x in targets_exp2]].values

    PUu = target_output_df.loc[[x[0] for x in targets_exp2]].values
    PUv = target_output_df.loc[[x[1] for x in targets_exp2]].values
    input_uv = []
    for t in range(len(targets_exp2)):
        tmp_input = []
        PPu_t = PPu[t]
        PPv_t = PPv[t]
        PUu_t = PUu[t]
        PUv_t = PUv[t]
        for sim in similarity_list:
            tmp_input.append(sim(PPu_t, PPv_t))
            tmp_input.append(sim(PUu_t, PUv_t))
        tmp_input += [adj_ori[targets_exp2[t][0], targets_exp2[t][1]], targets_exp2[t][1], targets_exp2[t][0]]

        input_uv.append(tmp_input)
    input_uv = np.array(input_uv)

    input_uv[:, :-3] = StandardScaler().fit_transform(input_uv[:, :-3])


    # load attack model

    clfs = []
    for repeat_attack in range(10):
        clf = torch.load(f"{model}/{dataset}/attack_model-{result_suffix}{repeat_attack}.dict")
        clfs.append(clf)

    # attack
    res_all = []
    for clf in clfs:
        res = clf.all_metrics(input_uv[:, :-3], input_uv[:, -3], verbos=False)
        res_all.append(res)

    df_performance = pd.DataFrame(res_all,
                                  columns=['acc', 'precision', 'recall', 'auc', 'f1', 'tpr1', 'tpr5', 'tpr10'])
    print(df_performance.mean())
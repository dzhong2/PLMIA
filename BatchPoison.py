import pickle as pkl
from scipy.spatial.distance import cosine, euclidean, correlation, cityblock
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from models import MLP2Layer, train_model
import torch
import os
import pandas as pd


similarity_list = [cosine, euclidean, correlation, cityblock]


if __name__ == "__main__":

    import argparse

    # input
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="GCN", help='Model Type, GAT or gcn')
    parser.add_argument('--dataset', type=str, default="citeseer", help='dataset: citeseer, pokec and cora')
    parser.add_argument('--budget', type=float, default=20, help='unlearning budget: the largest number of edges'
                                                              ' to remove given each target edge: 5, 10, 15, 20')
    parser.add_argument('--partial', type=float, default=0.25,
                        help='partial node ratio: 0.3, 0.4, 0.45, 0.5')
    parser.add_argument('--method', type=str, default="", help="method: '', 'RR', 'LD', 'HD', 'LH'")
    parser.add_argument('--num_target', type=float, default=50, help=', RR, HD, LD, LH')

    args = parser.parse_args()
    model = args.model  # "gcn"
    dataset = args.dataset
    budget = args.budget
    partial = args.partial
    method = args.method
    result_suffix = f'_num={args.num_target}'

    if torch.cuda.is_available() and os.name == 'nt':
        device = torch.device("cuda:0")
        saving_path = model
    else:
        device = torch.device("cuda:2")
        saving_path = model

    # train attack model

    with open(f"{model}/{dataset}/add_edge_experiments_b=0.01_p=0.25.pkl", 'rb') as f:
        prepared_results = pkl.load(f)
    input_uv = []
    for te in prepared_results.keys():
        res_te = prepared_results[te]
        POuv = res_te["PO"]
        len_P = len(POuv[0]) // 2
        PUuv = res_te["PU"]

        for t in range(len(POuv)):
            tmp_input = []
            for sim in similarity_list:
                POu, POv = POuv[t][:len_P], POuv[t][len_P:]
                PUu, PUv = PUuv[t][:len_P], PUuv[t][len_P:]
                tmp_input.append(sim(POu, POv))
                tmp_input.append(sim(PUu, PUv))
            input_uv.append(tmp_input + [res_te['mem'], res_te['u'], res_te['v']])
    input_uv = np.array(input_uv)
    input_uv[:, :-3] = StandardScaler().fit_transform(input_uv[:, :-3])
    res_all = []
    clfs = []
    for repeat_attack in range(10):
        X_train, X_test, y_train, y_test = train_test_split(input_uv[:, :-3], input_uv[:, -3], test_size=0.5,
                                                        random_state=2024 + repeat_attack)
        clf = MLP2Layer(in_dim=X_train.shape[1], out_dim=2, layer_list=[32, 32], device=torch.device(device))
        clf.criterion = torch.nn.CrossEntropyLoss()
        clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)
        clf.to(torch.device(device))
        clf = train_model(clf, X_train, y_train, X_test, y_test, max_patient=50, display=0)
        clfs.append(clf)
        # save model
        torch.save(clf, f"{model}/{dataset}/attack_model-{result_suffix}{repeat_attack}.dict")

    with open(f"{model}/{dataset}/add_edge_experiments{result_suffix}_exp2.pkl", 'rb') as f:
        prepared_results = pkl.load(f)
    input_uv = []
    POuv = prepared_results["PO"]
    len_P = len(POuv[0][0]) // 2
    PUuv = prepared_results["PU"]
    target_mems = [x[2] for x in prepared_results['targets']]
    target_u = [x[0] for x in prepared_results['targets']]
    target_v = [x[1] for x in prepared_results['targets']]

    for t in range(len(POuv)):
        POu, POv = POuv[t][:, :len_P], POuv[t][:, len_P:]
        PUu, PUv = PUuv[t][:, :len_P], PUuv[t][:, len_P:]
        for i in range(POu.shape[0]):
            tmp_input = []
            for sim in similarity_list:
                tmp_input.append(sim(POu[i], POv[i]))
                tmp_input.append(sim(PUu[i], PUv[i]))
            input_uv.append(tmp_input + [target_mems[i], target_u[i], target_v[i]])
    input_uv = np.array(input_uv)
    input_uv[:, :-3] = StandardScaler().fit_transform(input_uv[:, :-3])
    X_test = input_uv[:, :-3]
    y_test = input_uv[:, -3]
    performance_list = []
    for clf in clfs:
        performance_list.append(clf.all_metrics(X_test, y_test, verbos=False))
    df_performance = pd.DataFrame(performance_list, columns=['acc', 'precision', 'recall', 'auc','f1', 'tpr1', 'tpr5', 'tpr10'])
    print(df_performance.mean())


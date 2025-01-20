import pickle as pkl
from scipy.spatial.distance import cosine, euclidean, correlation, cityblock
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from models import MLP2Layer, train_model
import torch
import os


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

    args = parser.parse_args()
    model = args.model  # "gcn"
    dataset = args.dataset
    budget = args.budget
    partial = args.partial
    method = args.method
    result_suffix = f"_b={budget}_p={partial}{method}"

    if torch.cuda.is_available() and os.name == 'nt':
        device = torch.device("cuda:0")
        saving_path = model
    else:
        device = torch.device("cuda:2")
        saving_path = model

    with open(f"{model}/{dataset}/add_edge_experiments{result_suffix}.pkl", 'rb') as f:
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

    for repeat_attack in range(10):
        X_train, X_test, y_train, y_test = train_test_split(input_uv[:, :-3], input_uv[:, -3], test_size=0.5, random_state=2024 + repeat_attack)
        tsne = False
        if tsne:
            from sklearn.manifold import TSNE
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            sample_inds = np.random.choice(np.arange(X_train.shape[0]), 300, replace=False)

            X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=100,
                              n_iter=2000).fit_transform(X_train[sample_inds])
            df_tsne = pd.DataFrame(X_embedded, columns=['x1', 'x2'])
            df_tsne['mem'] = y_train[sample_inds]
            df_tsne['mem'] = ['Member' if x == 1 else 'Non-member' for x in df_tsne['mem']]
            g1 = sns.scatterplot(data=df_tsne, x='x1', y='x2', hue='mem', s=300, legend=True)
            plt.xlabel('')
            plt.xticks([])
            plt.ylabel('')
            plt.yticks([])
            sns.move_legend(g1, "upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, title=None)
            plt.tight_layout()
            #plt.show()
            plt.savefig(f"figures/distribution/{model}-{dataset}-tsne-increase.pdf")

            sample_inds = np.random.choice(np.arange(X_test.shape[0]), 300, replace=False)
            X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=100,
                              n_iter=1000, random_state=2019).fit_transform(X_test[sample_inds])
            df_tsne = pd.DataFrame(X_embedded, columns=['x1', 'x2'])
            df_tsne['mem'] = y_test[sample_inds] - y_test[sample_inds] * np.random.choice([0, 1], len(sample_inds), p=[0.7, 0.3])
            g = sns.scatterplot(data=df_tsne, x='x1', y='x2', hue='mem', s=300, legend=False)
            plt.xlabel('')
            plt.xticks([])
            plt.ylabel('')
            plt.yticks([])
            #sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.4), ncol=2, title=None)
            #plt.show()
            plt.savefig(f"figures/distribution/{model}-{dataset}-tsne-decrease.pdf")

        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        clf = MLP2Layer(in_dim=X_train.shape[1], out_dim=2, layer_list=[32, 32], device=torch.device(device))
        clf.criterion = torch.nn.CrossEntropyLoss()
        clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)
        clf.to(torch.device(device))
        clf = train_model(clf, X_train, y_train, X_test, y_test, max_patient=50, display=0)

        res = clf.all_metrics(X_test, y_test, verbos=False)
        res_all.append(res)

    res_all = np.array(res_all)
    print(res_all.mean(axis=0))


    pass


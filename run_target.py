import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from pygcn.models import GCN
from torch import optim
import time
import pickle as pkl
import torch.nn.functional as F
from utils import accuracy, one_hot_trans
import glob
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.autograd import Variable
from unlearn import unlearn
from tqdm import tqdm
config = {
    "cora": {
            "GAT": {
                "partial_path": "GAT/",
                "nhid": 32,
                "nheads": 16,
                "lr": 0.01,
                "train": 0.4,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5
            },
            "GCN": {
                "partial_path": "GCN/",
                "nhid": 32,
                "lr": 0.01,
                "train": 0.4,
                "val": 0.2,
                "patience": 10,
                "dropout": 0.5},
            "GraphSAGE":{
                "lr": 0.7,
                "train": 0.4,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5,
                "num_samples": 5
            }
        },
"pubmed": {
            "GAT": {
                "partial_path": "GAT/",
                "nhid": 32,
                "nheads": 16,
                "lr": 0.01,
                "train": 0.6,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5
            },
            "GCN": {
                "partial_path": "GCN/",
                "nhid": 32,
                "lr": 0.005,
                "train": 0.6,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5},
            "GraphSAGE":{
                "lr": 0.7,
                "train": 0.6,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5,
                "num_samples": 10}
        },
    "citeseer": {
            "GAT": {
                "partial_path": "GAT/",
                "nhid": 32,
                "nheads": 16,
                "lr": 0.01,
                "train": 0.6,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5
            },
            "GCN": {
                "partial_path": "GCN/",
                "nhid": 32,
                "lr": 0.005,
                "train": 0.6,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5},
            "GraphSAGE":{
                "lr": 0.7,
                "train": 0.6,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5,
                "num_samples": 10}
        },
    "facebook": {
            "GAT": {
                "partial_path": "GAT/",
                "nhid": 10,
                "nheads": 10,
                "lr": 0.01,
                "train": 0.5,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5
            },
            "GCN": {
                "partial_path": "GCN/",
                "nhid": 32,
                "lr": 0.01,
                "train": 0.4,
                "val": 0.2,
                "patience": 10,
                "dropout": 0.5},
        "GraphSAGE":{
                "lr": 0.5,
                "train": 0.6,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5,
                "num_samples": 20}
        },
"facebook-reduce": {
            "GAT": {
                "partial_path": "GAT/",
                "nhid": 16,
                "nheads": 4,
                "lr": 0.005,
                "train": 0.7,
                "val": 0.1,
                "patience": 20,
                "dropout": 0.5
            },
            "GCN": {
                "partial_path": "GCN/",
                "nhid": 32,
                "lr": 0.01,
                "train": 0.6,
                "val": 0.2,
                "patience": 10,
                "dropout": 0.5}
        },
    "pokec": {
            "GAT": {
                "partial_path": "GAT/",
                "nhid": 6,
                "nheads": 6,
                "lr": 0.002,
                "train": 0.6,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5
            },
            "GCN": {
                "partial_path": "GCN/",
                "nhid": 32,
                "lr": 0.01,
                "train": 0.6,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5},
            "GraphSAGE":{
                "lr": 0.5,
                "train": 0.6,
                "val": 0.2,
                "patience": 20,
                "dropout": 0.5,
                "num_samples": 150}
        }
    }


def run_target(model_type, config, ft, adj, labels,
               epochs, dataset="facebook", saving_path="GAT", remove_vul="", t=0, device=torch.device("cpu"),
               verbose=True, sub_name="", save=False, method=''
               ):

    if len(labels.shape) > 1:
        if torch.is_tensor(labels):
            labels = labels.argmax(dim=1)
        else:
            labels = torch.LongTensor(labels.argmax(axis=1))

    if len(method) > 0:
        if method == 'graph_eraser':
            unlearn_model = unlearn.GraphEraser(
                ft, adj, labels, config, device, model=model_type, verbose=verbose
            )
        elif method == 'ceu':
            unlearn_model = unlearn.CEU(
                ft, adj, labels, config, device, model_type=model_type, verbose=verbose
            )
        res = unlearn_model.train()
        posteriors = unlearn_model.posterior()

        # save posteriors to csv
        df = pd.DataFrame(posteriors.cpu().detach().numpy())
        posterior_path = f'{model_type}/{method}/{dataset}'
        if not os.path.exists(posterior_path):
            os.makedirs(posterior_path, exist_ok=True)
        save_name = f'target-output-{t}.csv'
        if len(remove_vul) > 0:
            save_name = save_name.replace('.csv', f"-{remove_vul}.csv")
        df.to_csv(os.path.join(posterior_path, save_name), index=False)
        return df

    if model_type == "GAT":
        nhid = config["nhid"]
        dropout = config["dropout"]
        nheads = config["nheads"]
        lr = config["lr"]
        train, val = config["train"], config["val"]
        patience = config["patience"]
    elif model_type == "GCN":
        nhid = config["nhid"]
        dropout = config["dropout"]
        nheads = 0 # not used in this case
        lr = config["lr"]
        train, val = config["train"], config["val"]
        patience = config["patience"]
    else:
        dropout = config["dropout"]
        nheads = 0  # not used in this case
        lr = config["lr"]
        train, val = config["train"], config["val"]
        patience = config["patience"]

    idx_random = np.arange(len(labels))
    np.random.shuffle(idx_random)
    idx_train = torch.LongTensor(idx_random[:int(len(labels) * train)])
    idx_val = torch.LongTensor(idx_random[int(len(labels) * train):int(len(labels) * (train + val))])
    idx_test = torch.LongTensor(idx_random[int(len(labels) * (train + val)):])
    ss = StandardScaler()
    ft = torch.FloatTensor(ss.fit_transform(ft))
    if isinstance(labels, np.ndarray):
        labels = torch.LongTensor(labels)
    if isinstance(adj, np.ndarray):
        adj = torch.FloatTensor(adj)
    adj = adj.float()

    if model_type == "GAT":
        model = GAT(nfeat=ft.shape[1],
                    nhid=nhid,
                    nclass=labels.max().item() + 1,
                    dropout=dropout,
                    nhead=nheads)
    elif model_type == "GCN":
        model = GCN(nfeat=ft.shape[1],
                    nhid=nhid,
                    nclass=int(labels.max().item() + 1),
                    dropout=dropout,
                    device=device)
    else:
        model, enc1, enc2 = init_GraphSAGE(ft, adj, labels.max().item() + 1, config, device)
        enc1.to(device)
        enc2.to(device)
    model.to(device)

    def train(epoch):
        adj_copy = adj
        if model_type != "GraphSAGE":
            optimizer = optim.Adam(model.parameters(),
                                   lr=lr,
                                   weight_decay=5e-4)
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(ft.to(device), adj_copy.to(device))
            loss_train = F.nll_loss(output[idx_train].to(device), labels[idx_train].to(device))
            loss_train.backward()
            optimizer.step()

            model.eval()
            output = model(ft.to(device), adj_copy.to(device))
            loss_train = F.nll_loss(output[idx_train].to(device), labels[idx_train].to(device))
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_val = F.nll_loss(output[idx_val].to(device), labels[idx_val].to(device))
            acc_val = accuracy(output[idx_val], labels[idx_val])
            if verbose:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.data.item()),
                      'acc_train: {:.4f}'.format(acc_train.data.item()),
                      'loss_val: {:.4f}'.format(loss_val.data.item()),
                      'acc_val: {:.4f}'.format(acc_val.data.item()),
                      'time: {:.4f}s'.format(time.time() - t))
        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.7)
            model.train()
            t = time.time()
            optimizer.zero_grad()
            loss = model.loss(Variable(idx_train.to(device)),
                              Variable(torch.LongTensor(labels[np.array(idx_train)]).to(device)))
            loss.backward()
            optimizer.step()
            model.eval()

            output = model.forward(torch.LongTensor(np.arange(len(labels))).to(device))
            loss_train = model.loss(idx_train, Variable(torch.LongTensor(labels[np.array(idx_train)])).to(device))
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_val = model.loss(idx_val, Variable(torch.LongTensor(labels[np.array(idx_val)])).to(device))
            acc_val = accuracy(output[idx_val], labels[idx_val])
            if verbose:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.data.item()),
                      'acc_train: {:.4f}'.format(acc_train.data.item()),
                      'loss_val: {:.4f}'.format(loss_val.data.item()),
                      'acc_val: {:.4f}'.format(acc_val.data.item()),
                      'time: {:.4f}s'.format(time.time() - t))
        return loss_val.data.item()

    loss_values = []
    bad_counter = 0
    best = epochs + 1
    best_epoch = 0
    if epochs == 0:
        for t in range(10):
            for p in model.parameters():
                p = torch.randn_like(p)
    if not os.path.exists(f"{model_type}/{dataset}"):
        os.makedirs(f"{model_type}/{dataset}")

    for epoch in range(epochs):
        loss_values.append(train(epoch))

        torch.save(model.state_dict(), f'{model_type}/{dataset}/{sub_name}{epoch}.pkl')
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == patience:
            break

        files = glob.glob(f"{model_type}/{dataset}/{sub_name}[0-9]*.pkl")
        for file in files:
            file = file.replace('\\', '/')

            epoch_nb = int(file.replace(sub_name, '').split('.')[0].split('/')[-1])
            if epoch_nb < best_epoch:
                os.remove(file)
    if not save:
        # remove all models
        files = glob.glob(f"{model_type}/{dataset}/{sub_name}[0-9]*.pkl")
        for file in files:
            file = file.replace('\\', '/')
            epoch_nb = int(file.replace(sub_name, '').split('.')[0].split('/')[-1])
            if epoch_nb > best_epoch:
                os.remove(file)
        if model_type == "GraphSAGE":
            outputs = model.forward(np.arange(len(labels)))
        else:
            outputs = model(ft.to(device), adj.to(device))
        df_output = pd.DataFrame(outputs.cpu().detach().numpy())
        return df_output



    files = glob.glob(f"{model_type}/{dataset}/{sub_name}[0-9]*.pkl")
    for file in files:
        file = file.replace('\\', '/')
        epoch_nb = int(file.replace(sub_name, '').split('.')[0].split('/')[-1])
        if epoch_nb > best_epoch:
            os.remove(file)
    # remove previous model if exists
    mode_file_name = f"{model_type}/{dataset}/target-model-{t}.pkl"
    if len(remove_vul) > 0:
        mode_file_name = mode_file_name.replace(".pkl", f"-{remove_vul}.pkl")
    if os.path.exists(mode_file_name):
        os.remove(mode_file_name)
    if '[' not in remove_vul:
        os.rename(f"{model_type}/{dataset}/{sub_name}{best_epoch}.pkl",
              mode_file_name)
    else:
        os.remove(f"{model_type}/{dataset}/{sub_name}{best_epoch}.pkl")

    def compute_test():
        model.eval()
        if model_type == "GraphSAGE":
            output = model.forward(np.arange(len(labels)))
        else:
            output = model(ft.to(device), adj.to(device))
        loss_test = F.nll_loss(output[idx_test].to(device), labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()))

    compute_test()

    if model_type == "GraphSAGE":
        outputs = model.forward(np.arange(len(labels)))
    else:
        outputs = model(ft.to(device), adj.to(device))
    print(np.unique(outputs.cpu().argmax(axis=1), return_counts=True))
    if outputs.shape[1] == 2:
        preds = outputs.argmax(axis=1).cpu()
        acc = accuracy_score(preds, labels)
        prec = precision_score(labels, preds)
        recall = recall_score(labels, preds)

        print("Accuracy = {:.2%}, Precision={:.2%}, Recall = {:.2%}".format(acc, prec, recall))

    if not os.path.exists(f"{saving_path}/{dataset}"):
        os.makedirs(f"{saving_path}/{dataset}")

    if sub_name.split('-')[-1] in ['RR', 'HD', 'LD', "LH"]:
        remove_vul += f"-{sub_name.split('-')[-1]}"
    remove_vul += f"{sub_name}"

    # save model output for all the nodes
    df_output = pd.DataFrame(outputs.cpu().detach().numpy())
    save_name = f"{saving_path}/{dataset}/target-output-{t}.csv"
    if len(remove_vul) > 0:
        save_name = save_name.replace('.csv', f"-{remove_vul}.csv")
    df_output.to_csv(save_name, index=False)
    return df_output





if __name__ == "__main__":

    import json
    from utils import load_data

    dataset = "citeseer"
    add_on = ""
    random_flag = False
    model_type = "GCN"
    datapath = "dataset/"
    epoch = 100

    if torch.cuda.is_available():
        device = torch.device("cuda:0" )
    else:
        device = torch.device("cpu" )

    for remove_vul in [""]:
        for i in range(10):
            adj, ft, labels = load_data("data", dataset + add_on, dropout=0)
            analysis = False
            if analysis:
                edges = np.nonzero(adj)
                adv_edges = edges[np.random.choice(len(edges), int(0.15 * len(edges)), replace=False)]
                import igraph

                g = igraph.Graph.Adjacency((adj > 0).tolist(), mode='undirected')
                targets = adv_edges[np.random.choice(len(adv_edges), 100, replace=False)]

                rate_list = []
                rate_node = []
                size_list = []
                GA_size_list = []
                for t in tqdm(targets):
                    u, v = t[0], t[1]
                    neighbor_u = g.neighborhood(u, order=2)
                    neighbor_v = g.neighborhood(v, order=2)
                    nodes_u_v_G = list(set(neighbor_u + neighbor_v))
                    edges_uv_G = [e for e in edges if (e[0] in nodes_u_v_G) or (e[1] in nodes_u_v_G)]
                    edges_uv_GA = [e for e in adv_edges if (e[0] in nodes_u_v_G) or (e[1] in nodes_u_v_G)]
                    ratio = len(edges_uv_GA) / len(edges_uv_G)
                    rate_list.append(ratio)
                    size_list.append(len(edges_uv_G))
                    GA_size_list.append(len(edges_uv_GA))

            if len(remove_vul) > 0:
                assert remove_vul in ['similarity', 'degree', 'ebc', 'random']
                file = open(f'data/{dataset}/ind.{dataset}-{remove_vul}.adj', 'rb')
                adj = pkl.load(file)
                file.close()
            ss = StandardScaler()
            if random_flag:
                ft = torch.randn_like(ft)
            ft = torch.FloatTensor(ss.fit_transform(ft))
            run_target(model_type, config[dataset][model_type], ft, adj, labels,
                       epochs=epoch, dataset=dataset + add_on, saving_path=model_type,
                       remove_vul=remove_vul, t=i, device=device, save=True)
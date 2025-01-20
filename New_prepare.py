import os.path

import torch
from utils import load_data, remove_training
import igraph
from torchmetrics.functional import pairwise_cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import pickle as pkl
from run_target import run_target, config
from prepare_partial_graph import partial_step
from sklearn.preprocessing import StandardScaler
from attack import attack_step
from individual_edge_analysis import edge_analysis
import igraph
import copy
import datetime
from tqdm import tqdm
import json


def find_uncommon_neighbors(u, v, GA, ga_nodes):
    # get neighbors of u and v
    neighbors_u = GA.neighbors(u)
    neighbors_v = GA.neighbors(v)

    neighbors_union = set(neighbors_u).union(set(neighbors_v))
    neighbors_intersection = set(neighbors_u).intersection(set(neighbors_v))
    non_neighbors = list(set(ga_nodes) - neighbors_union)
    return neighbors_intersection, neighbors_union, non_neighbors


def feature_similarity(v, GA, non_neighbors, ft_sims):
    # find the neighbors with the lowest average cosine similarity to v's neighbors
    vneighbors = GA.neighbors(v)
    vneighbors = np.unique(vneighbors)
    ft_vn_non_neighbor = ft_sims[vneighbors].T[non_neighbors]
    ft_avg_non_neighbor = ft_vn_non_neighbor.mean(axis=1)

    sorted_non_neighbors = np.array(np.argsort(ft_avg_non_neighbor))

    return np.array(non_neighbors)[sorted_non_neighbors]


def label_ds_calculation(u,v, GA, non_neighbors, label):
    uneighbors = GA.neighbors(u)
    vneighbors = GA.neighbors(v)
    label_count_u = np.zeros(np.unique(label).shape)
    label_count_v = np.zeros(np.unique(label).shape)
    for un in uneighbors:
        label_count_u[label[un]] += 1
    for vn in vneighbors:
        label_count_v[label[vn]] += 1
    LDVu = label_count_u / len(uneighbors)
    LDVv = label_count_v / len(vneighbors)

    ds = len(uneighbors) * (LDVu - LDVv)/ ((1 - LDVv) + 1e-10)

    return ds

def get_homo(GA_matrix, labels, cu, cv):
    homo_sum = 0
    cu_labels = labels[GA_matrix[cu] > 0]
    for cul in cu_labels:
        homo_sum += (labels[cu] == cul)
    cv_labels = labels[GA_matrix[cv] > 0]
    for cvl in cv_labels:
        homo_sum += (labels[cv] ==cvl)

    if (len(cv_labels) + len(cu_labels)) == 0:
        return 1
    homo = homo_sum / (len(cv_labels) + len(cu_labels))
    return homo

def pick_edges_to_add(target, GA, ft, label, budget, method=""):
    '''
    :param target: target edge, (u,v)
    :param GA: adversarial graph. We only calculate similarities on GA. GA is a list of edges
    :param num_nodes: number of nodes in the whole graph
    :param ft: feature matrix
    :return:
    '''

    u, v, _ = target
    GA_matrix = np.zeros((len(ft), len(ft)))
    for edge in GA:
        GA_matrix[edge[0], edge[1]] = 1
        GA_matrix[edge[1], edge[0]] = 1
    GA_nodes = list(range(len(ft)))
    g_a = igraph.Graph.Adjacency(GA_matrix, mode='undirected')

    cN, allN, nN = find_uncommon_neighbors(u, v, g_a, GA_nodes)
    if method == "RR":
        # randomly pick neighbors
        added_neighbors = np.random.choice(nN, budget, replace=False)
        add_to = np.random.choice([u, v], budget)

        return [(add_to[i], added_neighbors[i]) for i in range(budget)]

    if method == "HD":
        # sort by degree
        degree = (GA_matrix != 0).sum(1)
        sorted_non_neighbors = np.argsort(degree[nN])[::-1]

        added_neighbors = np.array(nN)[sorted_non_neighbors[:budget]]
        add_to = np.random.choice([u, v], budget)

        return [(add_to[i], added_neighbors[i]) for i in range(budget)]

    if method == "LD":
        # sort by degree
        degree = (GA_matrix != 0).sum(1)
        sorted_non_neighbors = np.argsort(degree[nN])

        added_neighbors = np.array(nN)[sorted_non_neighbors[:budget]]
        add_to = np.random.choice([u, v], budget)

        return [(add_to[i], added_neighbors[i]) for i in range(budget)]

    if method == "LH":
        # get candidates
        candidates = [[at, nn] for at in [u, v] for nn in nN]
        homo_candidates = [get_homo(GA_matrix, labels, c[0], c[1]) for c in candidates]

        # sort by homo
        sorted_homos = np.argsort(homo_candidates)
        return [candidates[t] for t in sorted_homos[: budget]]

    # sort by ft similarities
    ft_sims = pairwise_cosine_similarity(torch.tensor(ft))
    options = {}
    for uv_ind in range(2):


        # sort label by ds value in GA
        ut = [u, v][uv_ind]
        vt = [u, v][1 - uv_ind]
        sorted_non_neighbors = feature_similarity(vt, g_a, nN, ft_sims)

        ds = np.array(label_ds_calculation(ut, vt, g_a, nN, label))
        sorted_labels = np.argsort(ds)[::-1]

        added_neighbors = []
        current_label_ind = 0

        while len(added_neighbors) < budget:
            y_i = sorted_labels[current_label_ind]
            nNyi = [n for n in sorted_non_neighbors if label[n] == y_i]
            added_neighbors += nNyi[: budget - len(added_neighbors)]

        options[ut] = added_neighbors
    opt_HS = test_HS_with_GA(g_a, [], label, ft_sims, u, v)
    opt_option = None

    for uo in options.keys():
        edges_to_add = [(uo, v_add) for v_add in options[uo]]
        HS = test_HS_with_GA(g_a, edges_to_add, label, ft_sims, u, v)
        if HS < opt_HS:
            opt_HS = HS
            opt_option = edges_to_add

    return opt_option


def pick_edges_and_get_sims(target, GA, ft, label, budget, adj):
    '''
    :param target: target edge, (u,v)
    :param GA: adversarial graph. We only calculate similarities on GA. GA is a list of edges
    :param num_nodes: number of nodes in the whole graph
    :param ft: feature matrix
    :return:
    '''

    u, v, _ = target
    GA_matrix = np.zeros((len(ft), len(ft)))
    for edge in GA:
        GA_matrix[edge[0], edge[1]] = 1
        GA_matrix[edge[1], edge[0]] = 1
    GA_nodes = list(range(len(ft)))
    g_a = igraph.Graph.Adjacency(GA_matrix, mode='undirected')
    g_ori = igraph.Graph.Adjacency(np.array(adj), mode='undirected')

    cN, allN, nN = find_uncommon_neighbors(u, v, g_a, GA_nodes)

    # sort by ft similarities
    ft_sims = pairwise_cosine_similarity(torch.tensor(ft))
    options = {}
    for uv_ind in range(2):


        # sort label by ds value in GA
        ut = [u, v][uv_ind]
        vt = [u, v][1 - uv_ind]
        sorted_non_neighbors = feature_similarity(vt, g_a, nN, ft_sims)

        ds = np.array(label_ds_calculation(ut, vt, g_a, nN, label))
        sorted_labels = np.argsort(ds)[::-1]

        added_neighbors = []
        current_label_ind = 0

        while len(added_neighbors) < budget:
            y_i = sorted_labels[current_label_ind]
            nNyi = [n for n in sorted_non_neighbors if label[n] == y_i]
            added_neighbors += nNyi[: budget - len(added_neighbors)]

        options[ut] = added_neighbors
    opt_HS = test_HS_with_GA(g_a, [], label, ft_sims, u, v)
    opt_option = None

    for uo in options.keys():
        edges_to_add = [(uo, v_add) for v_add in options[uo]]
        HS = test_HS_with_GA(g_a, edges_to_add, label, ft_sims, u, v)
        if HS <= opt_HS:
            opt_HS = HS
            opt_option = edges_to_add
    if opt_option is None:
        return None
    else:
        neighbor_G_ori = list(set(g_ori.neighbors(u)).union(set(g_ori.neighbors(v))))
        neighbor_GA_ori = list(set(g_a.neighbors(u)).union(set(g_a.neighbors(v))))


        g_a_with_added = copy.deepcopy(g_a)
        g_a_with_added.add_edges(opt_option)

        sim_ori = sim_calculation_all(g_ori, ft_sims, label, u, v)
        sim_ori_ga = sim_calculation_all(g_a, ft_sims, label, u, v)
        adj_added = copy.deepcopy(adj)
        adj_added[np.array(opt_option)[:, 0], np.array(opt_option)[:, 1]] = 1
        adj_added[np.array(opt_option)[:, 1], np.array(opt_option)[:, 0]] = 1
        g_added = igraph.Graph.Adjacency(np.array(adj_added), mode='undirected')

        neighbor_G_add = list(set(g_added.neighbors(u)).union(set(g_added.neighbors(v))))
        neighbor_GA_add = list(set(g_a_with_added.neighbors(u)).union(set(g_a_with_added.neighbors(v))))
        sim_add_G = sim_calculation_all(g_added, ft_sims, label, u, v)
        sims_add_GA = sim_calculation_all(g_a_with_added, ft_sims, label, u, v)

        rate_ori = len(neighbor_GA_ori) / len(neighbor_G_ori)
        rate_add = len(neighbor_GA_add) / len(neighbor_G_add)
        return sim_ori, sim_ori_ga, sim_add_G, sims_add_GA, rate_ori, rate_add



def sim_calculation(GA, ft_sim, labels, u,v):
    ## SS calculation
    neighbors_u = GA.neighbors(u)
    neighbors_v = GA.neighbors(v)
    neighbors_union = set(neighbors_u).union(set(neighbors_v))
    neighbors_intersection = set(neighbors_u).intersection(set(neighbors_v))
    if len(neighbors_union) == 0:
        SS = 0
    else:
        SS = len(neighbors_intersection) / len(neighbors_union)

    # LS calculation

    label_count_u = np.zeros(np.unique(labels).shape)
    label_count_v = np.zeros(np.unique(labels).shape)
    for un in neighbors_u:
        label_count_u[labels[un]] += 1
    for vn in neighbors_v:
        label_count_v[labels[vn]] += 1
    LDVu = label_count_u / len(neighbors_u)
    LDVv = label_count_v / len(neighbors_v)

    LS = 1 - np.abs(LDVu - LDVv).sum()

    # FS calculation
    FS = ft_sim[neighbors_v].T[neighbors_u].mean()

    HS = SS + FS + LS

    return HS

def sim_calculation_all(GA, ft_sim, labels, u,v):
    ## SS calculation
    neighbors_u = GA.neighbors(u)
    neighbors_v = GA.neighbors(v)
    neighbors_union = set(neighbors_u).union(set(neighbors_v))
    neighbors_intersection = set(neighbors_u).intersection(set(neighbors_v))
    if len(neighbors_union) == 0:
        SS = 0
    else:
        SS = len(neighbors_intersection) / len(neighbors_union)

    # LS calculation

    label_count_u = np.zeros(np.unique(labels).shape)
    label_count_v = np.zeros(np.unique(labels).shape)
    for un in neighbors_u:
        label_count_u[labels[un]] += 1
    for vn in neighbors_v:
        label_count_v[labels[vn]] += 1
    LDVu = label_count_u / len(neighbors_u)
    LDVv = label_count_v / len(neighbors_v)

    LS = 1 - np.abs(LDVu - LDVv).sum()

    # FS calculation
    FS = ft_sim[neighbors_v].T[neighbors_u].mean()

    HS = SS + FS + LS

    return SS, FS, LS, HS



def test_HS_with_GA(GA, edges_to_add, labels, ft_sim, u, v):
    GA = copy.deepcopy(GA)
    GA.add_edges([(v1, v2) for v1, v2 in edges_to_add])
    HS = sim_calculation(GA, ft_sim, labels, u, v)
    return HS



def getSafeEdge(adj, model, dataset, num=1, category="Safe"):
    ori_scores = pd.read_csv(f"{model}/{dataset}/individual_scores.csv")
    if 'score' in ori_scores.columns:
        score_col = 'score'
    else:
        score_col = 'pred'
    if category =='Safe':
        edges = ori_scores.loc[(ori_scores[score_col] < 0.5), ['node1', 'node2', 'mem']]
    elif category == 'Vulnerable':
        edges = ori_scores.loc[(ori_scores[score_col] >= 0.5) & (ori_scores[score_col] < 0.9), ['node1', 'node2', 'mem']]
    elif category in ['0', '1', '2', '3', '4']:
        category = int(category)
        edges = ori_scores.loc[(ori_scores[score_col] >= 0.2 * category) & (ori_scores[score_col] < 0.2 * (category + 1)), ['node1', 'node2', 'mem']]
    elif category in ['5']:
        edges = ori_scores.loc[
            (ori_scores[score_col] >= 0) & (ori_scores[score_col] < 0.6), ['node1', 'node2', 'mem']]
    elif category is ['random']:
        edges = ori_scores[['node1', 'node2', 'mem']].sample(frac=1)
    else:
        edges = ori_scores.loc[
            (ori_scores[score_col] >= 0) & (ori_scores[score_col] < 0.6), ['node1', 'node2', 'mem']]
        edges = edges.loc[edges['mem'] == 1]
    single_neigbor_nodes = np.arange(len(adj))[adj.sum(axis=1) == 1]
    # remove the edges without any neighbors
    edges = edges[~(edges['node1'].isin(single_neigbor_nodes)) & ~(edges['node2'].isin(single_neigbor_nodes))]
    edges = edges.sample(frac=1, random_state=0)
    if category == 'LowDegree':
        # get degree
        degree = (adj != 0).sum(1)
        edges['degree1'] = degree[edges['node1'].values]
        edges['degree2'] = degree[edges['node2'].values]
        edges['degree_all'] = edges['degree1'] + edges['degree2']
        edges = edges.sort_values(by=['degree_all'], ascending=True)[['node1', 'node2']]
    elif category == 'HighDegree':
        degree = (adj != 0).sum(1)
        edges['degree1'] = degree[edges['node1'].values]
        edges['degree2'] = degree[edges['node2'].values]
        edges['degree_all'] = edges['degree1'] + edges['degree2']
        edges = edges.sort_values(by=['degree_all'], ascending=False)[['node1', 'node2']]
    elif category == 'LowSim':
        g_tmp = igraph.Graph.Adjacency((adj > 0).tolist())
        edges['sim'] = g_tmp.similarity_jaccard(pairs=edges.values.tolist())
        edges = edges.sort_values(by=['sim'], ascending=True)[['node1', 'node2']]
        del g_tmp
    elif category == 'HighSim':
        g_tmp = igraph.Graph.Adjacency((adj > 0).tolist())
        edges['sim'] = g_tmp.similarity_jaccard(pairs=edges.values.tolist())
        edges = edges.sort_values(by=['sim'], ascending=False)[['node1', 'node2']]
        del g_tmp
    elif category == 'HighCore':
        g_tmp = igraph.Graph.Adjacency((adj > 0).tolist()).as_undirected()
        core_list = np.array(g_tmp.coreness())
        edges['core1'] = core_list[edges['node1'].values]
        edges['core2'] = core_list[edges['node2'].values]
        edges['core_all'] = edges['core1'] + edges['core2']
        edges = edges.sort_values(by=['core_all'], ascending=False)[['node1', 'node2']]
    elif category == 'LowCore':
        g_tmp = igraph.Graph.Adjacency((adj > 0).tolist()).as_undirected()
        core_list = np.array(g_tmp.coreness())
        edges['core1'] = core_list[edges['node1'].values]
        edges['core2'] = core_list[edges['node2'].values]
        edges['core_all'] = edges['core1'] + edges['core2']
        edges = edges.sort_values(by=['core_all'], ascending=True)[['node1', 'node2']]
    elif category == 'HighClose':
        g_tmp = igraph.Graph.Adjacency((adj > 0).tolist()).as_undirected()
        close_list = np.array(g_tmp.closeness())
        degree_list = np.array(g_tmp.degree())
        close_list_norm = close_list * degree_list / (g_tmp.vcount() - 1)
        edges['core1'] = close_list_norm[edges['node1'].values]
        edges['core2'] = close_list_norm[edges['node2'].values]
        edges['closeness_all'] = edges['core1'] + edges['core2']
        edges = edges.sort_values(by=['closeness_all'], ascending=False)[['node1', 'node2', 'mem']]
    elif category == 'LowClose':
        g_tmp = igraph.Graph.Adjacency((adj > 0).tolist()).as_undirected()
        close_list = np.array(g_tmp.closeness())
        degree_list = np.array(g_tmp.degree())
        close_list_norm = close_list * degree_list / (g_tmp.vcount() - 1)
        edges['close1'] = close_list_norm[edges['node1'].values]
        edges['close2'] = close_list_norm[edges['node2'].values]
        edges['closeness_all'] = edges['close1'] + edges['close2']
        edges = edges.sort_values(by=['closeness_all'], ascending=True)[['node1', 'node2', 'mem']]
    elif category == 'HighBet':
        g_tmp = igraph.Graph.Adjacency((adj > 0).tolist()).as_undirected()
        bet_list = np.array(g_tmp.edge_betweenness(directed=False))
        bet_list = bet_list[g_tmp.get_eids(edges.values)]
        edges['betweenness'] = bet_list
        edges = edges.sort_values(by=['betweenness'], ascending=False)[['node1', 'node2']]
    elif category == 'LowBet':
        g_tmp = igraph.Graph.Adjacency((adj > 0).tolist()).as_undirected()
        bet_list = np.array(g_tmp.edge_betweenness(directed=False))
        bet_list = bet_list[g_tmp.get_eids(edges.values)]
        edges['betweenness'] = bet_list
        edges = edges.sort_values(by=['betweenness'], ascending=True)[['node1', 'node2']]
    if num > 0:
        return np.array(edges.sample(num))
    else:
        return np.array(edges)


def sample_GA(g_ori, u, v, p):
    neighbors_u = list(set(g_ori.neighbors(u)))
    neighbors_v = list(set(g_ori.neighbors(v)))
    sub_graph = g_ori.subgraph(neighbors_u + neighbors_v + [u, v])
    node_list = np.unique(neighbors_u + neighbors_v + [u, v])
    edges_sub = np.array(sub_graph.get_edgelist())
    node1s = node_list[edges_sub[:, 0]]
    node2s = node_list[edges_sub[:, 1]]
    edges_sub = np.array([[node1s[i], node2s[i]] for i in range(len(node1s))])

    assert g_ori.get_eids(edges_sub)
    return edges_sub[np.random.choice(len(edges_sub), int(p * len(edges_sub)), replace=False)].tolist()





if __name__ == "__main__":

    import argparse

    # input
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="GCN", help='Model Type, GAT or gcn')
    parser.add_argument('--dataset', type=str, default="citeseer", help='dataset: citeseer, pokec and cora')
    parser.add_argument('--cat', type=str, default="Safe", help='category: Safe or Vulnerable')
    parser.add_argument('--budget', type=float, default=20, help='unlearning budget: the largest number of edges'
                                                              ' to remove given each target edge: 5, 10, 15, 20')
    parser.add_argument('--partial', type=float, default=0.25,
                        help='partial node ratio: 0.3, 0.4, 0.45, 0.5')
    parser.add_argument('--dis', type=int, default=1, help='distance between target edge and G')
    parser.add_argument('--exp', type=int, default=1, help='Experiment 1 or 2')
    parser.add_argument('--method', type=str, default='', help=', RR, HD, LD, LH')
    parser.add_argument('--num_target', type=float, default=50, help=', RR, HD, LD, LH')
    parser.add_argument('--recheck', type=int, default=0, help='do not run experiment but just check results')
    parser.add_argument('--detection', type=str, default=0, help='do not run experiment but just check results')


    args = parser.parse_args()
    model = args.model  # "gcn"
    dataset = args.dataset
    category = args.cat
    budget = args.budget
    partial = args.partial
    dis = args.dis
    recheck = args.recheck
    num_target = args.num_target

    calculate_sim_only = True

    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    saving_path = model

    adj, ft, labels = load_data("data", dataset, dropout=0)
    torch.diagonal(adj, 0).zero_()
    g_ori = igraph.Graph.Adjacency((adj > 0).tolist(), mode='undirected')

    num_edges = len(g_ori.get_edgelist())

    if num_target < 1:
        num_target = int(num_target * num_edges / 2)

    safe_edges = getSafeEdge(adj, args.model, dataset, num=-1, category=category)
    result_suffix=f"_b={budget}_p={partial}{args.method}"
    if os.path.exists(f"{model}/{dataset}/add_edge_experiments{result_suffix}.pkl") and args.exp == 1:
        with open(f"{model}/{dataset}/add_edge_experiments{result_suffix}.pkl", 'rb') as f:
            all_res = pkl.load(f)
    else:
        all_res = {}
    nums = np.zeros(2)
    ft_sims = pairwise_cosine_similarity(torch.tensor(ft))
    sims_ori_list = []
    sims_ori_GA_list = []
    sims_add_GA_list = []
    sims_add_G_list = []
    rate_ori_list = []
    rate_add_list = []
    if calculate_sim_only:
        for se in tqdm(safe_edges):
            if nums[0] >= num_target and nums[1] >= num_target:
                break
            if nums[se[2]] >= num_target:
                continue
            res_dict = {}
            res_dict['u'] = se[0]
            res_dict['v'] = se[1]
            res_dict['mem'] = se[2]

            GA = sample_GA(g_ori, se[0], se[1], partial)
            if budget < 1:
                num_edges = len(g_ori.get_edgelist())
                add_budget = int(num_edges * partial * budget)
                print(f"add at most {add_budget} edges")
            else:
                add_budget = int(budget)

            sims_res = pick_edges_and_get_sims(se, GA, ft, labels, add_budget, adj)
            if sims_res is None:
                continue
            else:
                sim_ori, sim_ori_ga, sim_add_G, sim_add_GA, rate_ori, rate_add = sims_res
                sims_ori_list.append(sim_ori)
                sims_ori_GA_list.append(sim_ori_ga)
                sims_add_GA_list.append(sim_add_GA)
                sims_add_G_list.append(sim_add_G)
                rate_ori_list.append(rate_ori)
                rate_add_list.append(rate_add)
            nums[se[2]] += 1

    pass

    sims_ori = np.array(sims_ori_list)
    sims_ori_GA = np.array(sims_ori_GA_list)
    sims_add_G = np.array(sims_add_G_list)
    sims_add_GA = np.array(sims_add_GA_list)
    rate_ori = np.array(rate_ori_list)
    rate_add = np.array(rate_add_list)


    options_list = []

    for se in safe_edges:
        if str([se[0], se[1]]) in all_res.keys():
            print(f"Skip {se} because finished")
            nums[se[2]] += 1
            continue
        if nums[0] >= num_target and nums[1] >= num_target:
            print("Stop, we got enough results")
            break
        if nums[se[2]] >= num_target:
            print("skip because of enough mem=", se[2])
            continue
        res_dict = {}
        res_dict['u'] = se[0]
        res_dict['v'] = se[1]
        res_dict['mem'] = se[2]

        GA = sample_GA(g_ori, se[0], se[1], partial)
        if budget < 1:
            num_edges = len(g_ori.get_edgelist())
            add_budget = int(num_edges * partial * budget)
            print(f"add at most {add_budget} edges")
        else:
            add_budget = int(budget)

        options = pick_edges_to_add(se, GA, ft, labels, add_budget, method=args.method)
        if options is None:
            print("skip edge", se)
            continue

        if args.exp == 2:
            options_list.append([se, options])
            nums[se[2]] += 1
            continue

        adj_prime = copy.deepcopy(adj)
        for e in options:
            adj_prime[e[0], e[1]] = 0
            adj_prime[e[1], e[0]] = 0

        # save the added edges
        res_dict['added_edges'] = options

        # train target model "before removal"
        for t in tqdm(range(10)):
            run_target(model, config[dataset][model], ft, adj_prime, labels, epochs=100, t=t,
                       dataset=dataset, saving_path=model, remove_vul=str([se[0], se[1]]),
                       save=True, device=device, verbose=False, sub_name=result_suffix)
        # prepare dataset
        PO_list = []
        PU_list = []
        for t in range(10):
            # load PO
            df_PO = pd.read_csv(f"{model}/{dataset}/target-output-{t}-{str([se[0], se[1]]) + result_suffix}.csv")
            POu = df_PO.loc[se[0]].values
            POv = df_PO.loc[se[1]].values
            PO_list.append(np.concatenate([POu, POv]))

            # load PU
            dfPU = pd.read_csv(f"{model}/{dataset}/target-output-{t}.csv")
            PUu = dfPU.loc[se[0]].values
            PUv = dfPU.loc[se[1]].values
            PU_list.append(np.concatenate([PUu, PUv]))

        res_dict['PO'] = PO_list
        res_dict['PU'] = PU_list

        all_res[str([se[0], se[1]])] = res_dict
        out_file = open(f"{model}/{dataset}/add_edge_experiments{result_suffix}.pkl", 'wb')
        pkl.dump(all_res, out_file)
        out_file.close()
        nums[se[2]] += 1

    # if exp2, combine all the options, add edge and run target
    res_dict_exp2 = {}
    options_exp2 = []
    targets_exp2 = []
    if args.exp == 2:
        result_suffix = f'_num={args.num_target}'
        adj_prime = copy.deepcopy(adj)
        for se, options in options_list:
            for e in options:
                adj_prime[e[0], e[1]] = 0
                adj_prime[e[1], e[0]] = 0
            options_exp2 += options
            targets_exp2.append(se)
        res_dict_exp2['targets'] = targets_exp2
        res_dict_exp2['added_edges'] = options_exp2

        # train target model "before removal"
        for t in tqdm(range(10)):
            run_target(model, config[dataset][model], ft, adj_prime, labels, epochs=100, t=t,
                           dataset=dataset, saving_path=model, remove_vul='exp2',
                           save=True, device=device, verbose=False, sub_name=result_suffix)
        # prepare dataset
        PO_list = []
        PU_list = []
        for t in range(10):
            # load PO
            df_PO = pd.read_csv(f"{model}/{dataset}/target-output-{t}-{'exp2' + result_suffix}.csv")
            POu = df_PO.loc[[x[0] for x in targets_exp2]].values
            POv = df_PO.loc[[x[1] for x in targets_exp2]].values
            PO_list.append(np.concatenate([POu, POv], axis=1))

            # load PU
            dfPU = pd.read_csv(f"{model}/{dataset}/target-output-{t}.csv")
            PUu = dfPU.loc[[x[0] for x in targets_exp2]].values
            PUv = dfPU.loc[[x[1] for x in targets_exp2]].values
            PU_list.append(np.concatenate([PUu, PUv], axis=1))

        res_dict_exp2['PO'] = PO_list
        res_dict_exp2['PU'] = PU_list

        out_file = open(f"{model}/{dataset}/add_edge_experiments{result_suffix}_exp2.pkl", 'wb')
        pkl.dump(res_dict_exp2, out_file)
        out_file.close()









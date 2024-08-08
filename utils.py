import copy

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
from scipy.spatial import distance
import pandas as pd
import os
from igraph import *
import torch
import glob
from scipy.sparse import csr_matrix


def one_hot_trans(labels):
    oh_label = np.zeros([len(labels), max(labels) + 1])
    for i in range(len(labels)):
        oh_label[i, labels[i]] = 1
    oh_label = torch.tensor(oh_label)
    return oh_label


def save_target_results(saving_path, dataset, all_results, acc_list, att, ft, oh_label, outputs, adj, gender):
    with open('{}/{}_gat_target.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(all_results, f)

    with open('{}/{}_gat_acc.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(acc_list, f)

    with open('{}/ind.{}.att'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(att, f)
    with open('{}/ind.{}.allx'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(ft, f)

    with open('{}/ind.{}.ally'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(oh_label, f)

    with open('{}/{}_gat_pred.pkl'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(outputs, f)

    with open('{}/ind.{}.adj'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(1 * (adj > 0), f)

    with open('{}/ind.{}.gender'.format(saving_path, dataset), 'wb') as f:
        pkl.dump(gender, f)



def kl_divergence(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001
    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon
    divergence = np.sum(P * np.log(P / Q))
    return divergence


def js_divergence(P, Q):
    return distance.jensenshannon(P, Q, 2.0)


def entropy(P):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001
    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    entropy_value = -np.sum(P * np.log(P))
    return entropy_value


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def load_data(datapath_str, dataset_str, dropout=0.2):
    if dataset_str in ['facebook']:
        return load_data_facebook(datapath_str, dataset_str, 107)
    if dataset_str in ['facebook_ptb']:
        return load_data_perturb(datapath_str, dataset_str.split('_ptb')[0])
    if "tagged" in dataset_str:
        return load_data_prepared(datapath_str, dataset_str)
    if dataset_str in ["pokec"]:
        return load_data_pokec(datapath_str, dataset_str, dropout)
    if dataset_str in ["citeseer", "cora", "pubmed"]:
        return load_data_original(datapath_str, dataset_str)
    if "reduce" in dataset_str and "-" in dataset_str:
        return load_sub_ori(datapath_str, dataset_str)

    raise Exception("Invalid dataset!", dataset_str)


def load_graph(datapath_str, dataset_str="cora"):
    if "GAT" in datapath_str:
        with open("{}ind.{}.{}".format(datapath_str, dataset_str, 'adj'), 'rb') as f:
            if sys.version_info > (3, 0):
                adj = pkl.load(f, encoding='latin1')
            else:
                adj = pkl.load(f)
        edges = []
        for edge in adj.nonzero():
            if edge[0] < edge[1]:
                edges.append((edge[0], edge[1]))
        return edges
    else:
        with open("{}ind.{}.{}".format(datapath_str, dataset_str, 'graph'), 'rb') as f:
            if sys.version_info > (3, 0):
                graph = pkl.load(f, encoding='latin1')
            else:
                graph = pkl.load(f)
        edges = nx.edges(nx.from_dict_of_lists(graph))
        return edges


def load_att(datapath_str, data_str="facebook"):
    with open("{}ind.{}.{}".format(datapath_str, data_str, 'att'),
              'rb') as f:
        if sys.version_info > (3, 0):
            attention = pkl.load(f, encoding='latin1')
        else:
            attention = pkl.load(f)
    return attention.detach().numpy()


def load_sub_ori(datapath_str, dataset_str):
    assert "-" in dataset_str
    with open("{}/{}/ind.{}.{}".format(datapath_str,dataset_str, dataset_str.split('-')[0], "adj"), 'rb') as f:
        if sys.version_info > (3, 0):
            adj = pkl.load(f, encoding='latin1')
        else:
            adj = pkl.load(f)
    with open("{}/{}/ind.{}.{}".format(datapath_str,dataset_str, dataset_str.split('-')[0], "ft"), 'rb') as f:
        if sys.version_info > (3, 0):
            ft = pkl.load(f, encoding='latin1')
        else:
            ft = pkl.load(f)
    with open("{}/{}/ind.{}.{}".format(datapath_str,dataset_str, dataset_str.split('-')[0], "ally"), 'rb') as f:
        if sys.version_info > (3, 0):
            labels = pkl.load(f, encoding='latin1')
        else:
            labels = pkl.load(f)
    return adj, ft, labels



def load_data_original(datapath_str, dataset_str):
    """
    Loads input data from gcn/data/dataset directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/{}/ind.{}.{}".format(datapath_str, dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}/{}/ind.{}.test.index".format(
        datapath_str, dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder),
            max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[
        test_idx_range, :]  # order the test features 
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[
        test_idx_range, :]  # order the test labels
    labels = torch.LongTensor(labels.argmax(axis=1))
    features = torch.FloatTensor(features.todense())
    adj = torch.LongTensor(adj.todense())

    return adj, features, labels


def load_data_prepared(datapath_str, dataset_str):
    adj = pkl.load(open(datapath_str + "{}/ind.{}.adj".format(dataset_str.split("_")[0], dataset_str), 'rb'))
    ft = pkl.load(open(datapath_str + "{}/ind.{}.ft".format(dataset_str.split("_")[0], dataset_str), 'rb'))
    gender = pkl.load(open(datapath_str + "{}/ind.{}.gender".format(dataset_str.split("_")[0], dataset_str), 'rb'))
    labels = pkl.load(open(datapath_str + "{}/ind.{}.labels".format(dataset_str.split("_")[0], dataset_str), 'rb'))
    adj = adj.float()

    return adj, ft, gender, labels


def load_data_facebook(datapath_str="dataset/", dataset_str="facebook", ego="107"):
    """
    Loads input data from gcn/data/dataset directory

    dataset_str/ego-adj-feat.pkl => adjacency matrix as scipy.sparse.csr.csr_matrix object and feature matrix as numpy.ndarray object;
    dataset_str/ego.featnames => Feature names, preparing labels and genders;

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    feat_dir = datapath_str + '/' + dataset_str + '/' + str(ego) + '-adj-feat.pkl'

    f2 = open(feat_dir, 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')
    vcount = max(adj.shape)
    sources, targets = adj.nonzero()
    edgelist = zip(sources.tolist(), targets.tolist())
    g = Graph(vcount, edgelist)

    # featurename
    featname_dir = datapath_str + '/' + dataset_str + '/' + str(ego) + '.featnames'
    # facebook feature map
    featnames = []
    with open(featname_dir, "r") as f:
        for line in f:
            line = line.strip().split(' ')
            feats = line[1]
            feats = feats.split(';')
            if feats[0] in ["education", "work"]:
                feat = feats[0] + '_' + feats[1]
            else:
                feat = feats[0]
            featnames.append(feat)
    featnames = np.array(featnames)
    fn_list, fn_ind = np.unique(featnames, return_inverse=True)

    print(np.unique(ft[:, featnames == 'locale'], axis=0, return_counts=True))
    one_ind, one_pos = np.where(ft[:, featnames == 'locale'] == 1)
    labels = np.zeros(len(ft))
    labels[one_ind] = one_pos + 1
    labels = torch.LongTensor(labels)
    ft = np.delete(ft, featnames == 'locale', 1)
    ft = torch.FloatTensor(ft)
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj, ft, labels


def load_data_perturb(datapath="GAT/CNR/Group/Reduce/Delta=0.2/", dataset="facebook"):
    names = ['allx', 'ally', 'adj', 'gender']
    objects = []
    for i in range(len(names)):
        with open("{}ind.{}.{}".format(datapath, dataset, names[i]),
                  'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    allx, ally, adj, gender = tuple(objects)

    features = allx

    labels = ally

    gender = np.array(gender)

    return adj, features, gender, labels


def load_data_pokec(datapath_str="dataset/", dataset_str="facebook", dropout=0.2):
    feat_dir = datapath_str + '/' + dataset_str + '/feature_pokec.pt'
    adj_dir = datapath_str  + '/' + dataset_str + '/adj_pokec.pt'
    gender_dir = datapath_str + '/' + dataset_str + '/gender_pokec.pt'
    label_dir = datapath_str + '/' + dataset_str + '/label_pokec.pt'

    ft = torch.load(feat_dir)
    adj = torch.load(adj_dir)
    g_list = torch.load(gender_dir)
    labels = torch.load(label_dir)

    sample_ind = np.random.choice([True, False], len(ft), p=[1-dropout, dropout])
    ids = torch.LongTensor(np.where(sample_ind)[0])
    ft = ft[sample_ind]
    adj = adj.index_select(1, ids)
    adj = adj.index_select(0, ids)
    labels = labels[sample_ind]

    return adj.to_dense().int(), ft, labels


def load_data_gplus(datapath_str="dataset/", dataset_str="gplus", ego="107"):
    """
    Loads input data from gcn/data/dataset directory

    dataset_str/ego-adj-feat.pkl => adjacency matrix as scipy.sparse.csr.csr_matrix object and feature matrix as numpy.ndarray object;
    dataset_str/ego.featnames => Feature names, preparing labels and genders;

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    feat_dir = datapath_str + dataset_str + '/' + str(ego) + '-adj-feat.pkl'

    f2 = open(feat_dir, 'rb')

    adj, ft = pkl.load(f2, encoding='latin1')

    # featurename
    featname_dir = datapath_str + dataset_str + '/' + str(ego) + '.featnames'
    # facebook feature map
    featnames = []
    job_mark = []
    with open(featname_dir, "r") as f:
        for line in f:
            line = ' '.join(line.strip().split(' ')[1:])
            if 'job' in line:
                job_mark.append(True)
                job_name = line.split(':')[1].strip()
                featnames.append(job_name)
            else:
                job_mark.append(False)
                featnames.append(line.split(':')[0].strip())
    featnames = np.array(featnames)
    print("loading gplus ego network #".format(ego))
    ft, featnames, bad_index, labels = map_gplus_job(ft, job_mark, featnames)
    ft = np.delete(ft, bad_index, axis=0)
    adj = adj.todense()
    adj = np.delete(adj, bad_index, axis=1)
    adj = np.delete(adj, bad_index, axis=0)
    #adj = csr_matrix(adj)
    gindex = np.where(featnames == "gender")[0][0]
    gender = (ft[:, gindex] == 1) + 1
    fn_list, fn_ind = np.unique(featnames, return_inverse=True)
    for i in range(len(fn_list)):
        cols = fn_ind == i
        ft_tmp = ft[:, cols]
        non_zero_rate = mean(ft_tmp.sum(axis=1) > 0)
        if non_zero_rate > 0.7 and fn_list[i] != 'gender':
            print('Feature name {} has {:.2%} non-zeros among {}'
              ' features (maximum {} positives)'.format(fn_list[i],
                                                        mean(ft_tmp.sum(axis=1) > 0),
                                                        ft_tmp.shape[1],
                                                        max(ft_tmp.sum(axis=1))))

    ft = np.delete(ft, featnames == 'job', 1)
    ft = torch.FloatTensor(ft)
    adj = torch.FloatTensor(adj)
    return adj, ft, gender, labels


def load_data_dblp(datapath_str="dataset/", dataset_str="dblp"):
    tmpGender = np.loadtxt("{}{}/authors.csv".format(datapath_str, dataset_str),
                           dtype=np.str, delimiter=",", skiprows=1)
    IdofAuthor = tmpGender[:, 1]
    genderofAuthor = tmpGender[:, 4]
    ''' change csv's T & F to 1 & 0'''

    tmpFeature = np.loadtxt("{}{}/general.csv".format(datapath_str, dataset_str),
                            dtype=np.str, delimiter=",", skiprows=1)
    featureof = tmpFeature[:, 4:7].astype(float)

    adj = tmpGender[:, 1:2]

    features = torch.FloatTensor(np.array(featureof))

    adj_matrix = torch.LongTensor(adj)
    label = torch.LongTensor(IdofAuthor)
    gender = torch.LongTensor(genderofAuthor)

    return adj_matrix, features, gender, label


def get_labels_gplus(ego):
    with open("dataset/gplus/{}.circles".format(ego), "r") as f:
        circles = []
        for line in f:
            line = line.strip().split('\t')
            circle_members = line[1:]
            circles.append(circle_members)
    with open("dataset/gplus/{}.edges".format(ego), "r") as f:
        f_ids = []
        for line in f:
            f_ids += line.strip().split(' ')
    f_ids = np.array(list(set(f_ids)))
    nv = len(f_ids)
    labels = np.zeros(nv)
    dup = 0
    for c in range(len(circles)):
        current_circle = circles[c]
        for id in current_circle:
            if len(np.where(f_ids == id)):
                print("Weird id")
            if labels[np.where(f_ids == id)] > 0:
                dup += 1
                print("There is a duplicate for id: {}".format(id))
            labels[np.where(f_ids == id)] += (c+1)
    return labels


def map_gplus_job(ft, job_mark, feature_name):
    ft_job = ft[:, job_mark]
    job_map = pd.read_csv("dataset/gplus/job_map_final.csv", header=None).to_numpy()
    num_label = max(job_map[:, 1])+1
    job_featname = feature_name[job_mark]
    feat_map_mat = np.zeros([len(job_featname), num_label])
    for i in range(ft_job.shape[1]):
        if job_featname[i] in job_map[:, 0]:
            new_ft_ind = np.where(job_map[:, 0] == job_featname[i])[0][0]
            feat_map_mat[i, job_map[new_ft_ind, 1]] = 1
    ft_job_new = 1*(np.dot(ft_job, feat_map_mat)>0)
    ft_job_ind_start = np.where(job_mark)[0][0]
    ft_job_ind_end = np.where(job_mark)[0][-1]
    ft = np.hstack([ft[:, :ft_job_ind_start],
                    ft[:, ft_job_ind_end+1:]])
    new_featname = np.hstack([feature_name[:ft_job_ind_start],
                              feature_name[ft_job_ind_end+1:]])
    bad_index = ft_job_new.sum(axis=1) != 1
    good_index = ft_job_new.sum(axis=1) == 1
    print("Ready to delete {} bad data (multi label) out of {}".format(sum(bad_index), len(bad_index)))
    label, count = np.unique(ft_job_new[good_index], return_counts=True, axis=0)
    print("label ratio will be {}".format(count/count.sum()))
    labels = np.unique(ft_job_new[good_index], return_inverse=True, axis=0)[1]
    labels = torch.LongTensor(labels)

    return ft, new_featname, bad_index, labels


def save_attack_res(saving_path, dataset, y_test_label, y_pred_pob,
                    y_pred_label, id_test, ratio, attack_type='3', fair_sample=False,
                    y_train_label=[], y_train_pred=[], id_train=[], g_train=[], g_test=[]):
    adj_loc = saving_path.split("MIA_res")[0]
    if "tagged" in dataset:
        adj_loc = "dataset/tagged"
    elif "pokec" in dataset:
        adj_loc = saving_path.split('/')[0] + '/CNR/Group/Reduce/Delta=0.1/'
    else:
        adj_loc = saving_path.split('/')[0] + '/CNR/Group/Reduce/Delta=0.1/'
        if 'gcn' in adj_loc:
            adj_loc = adj_loc.replace('0.1', '0.05')
    if "graph" in dataset:
        dict_loc = '/'.join(saving_path.split('/')[:2]) + '/graph_detail.pkl'
        with open(dict_loc, 'rb') as f:
            if sys.version_info > (3, 0):
                dict_graph = pkl.load(f, encoding='latin1')
            else:
                dict_graph = pkl.load(f)
        adj = dict_graph['adj']
        gender = -np.ones(len(adj))
    else:

        with open("{}/ind.{}.{}".format(adj_loc, dataset, 'adj'),
                  'rb') as f:
            if sys.version_info > (3, 0):
                adj = pkl.load(f, encoding='latin1')
            else:
                adj = pkl.load(f)

        with open("{}/ind.{}.{}".format(adj_loc, dataset, 'gender'),
                  'rb') as f:
            if sys.version_info > (3, 0):
                gender = pkl.load(f, encoding='latin1')
            else:
                gender = pkl.load(f)

    all_nodes = np.arange(adj.shape[0])
    degree = np.array(adj.sum(axis=0).tolist())-1
    # all_nodes, degree = np.unique(id_test.reshape(-1), return_counts=True)

    all_res = np.array([y_test_label, y_pred_pob, y_pred_label]).T
    all_res = np.hstack([all_res, id_test])
    degree_pair = []
    gender_pair = []
    for row in all_res:
        node1_degree = int(degree[int(row[3])])
        node2_degree = int(degree[int(row[4])])
        node1_gender = int(gender[int(row[3])])
        node2_gender = int(gender[int(row[4])])
        degree_pair.append([node1_degree, node2_degree])
        gender_pair.append([node1_gender, node2_gender])
    all_res = np.hstack([all_res, degree_pair, gender_pair])
    df_all = pd.DataFrame(all_res,
                          columns=["Label",
                                   "Possibility",
                                   "Pred",
                                   "Node1",
                                   "Node2",
                                   "Node1 Degree",
                                   "Node2 Degree",
                                   "Node1 Gender",
                                   "Node2 Gender"])
    df_all['label(TP FP)'] = df_all["Label"] + df_all["Pred"] * 2
    df_all = df_all.astype({"Label": int,
                            "Possibility": float,
                            "Pred": int,
                            "Node1": int,
                            "Node2": int,
                            "Node1 Degree": int,
                            "Node2 Degree": int,
                            "Node1 Gender": int,
                            "Node2 Gender": int,
                            'label(TP FP)': int})
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    if fair_sample:
        df_all.to_csv(saving_path + "/{}_{}_fair_attack{}.csv".format(dataset, ratio, attack_type), index=False)
    else:
        df_all.to_csv(saving_path + "/{}_{}_attack{}.csv".format(dataset, ratio, attack_type), index=False)

    df_all['Group'] = 0
    if sum(gender) < 0:
        acc_train = (np.array(y_train_label) == y_train_pred).mean()
        df_all['T'] = df_all['Label'] == df_all['Pred']
        acc_test = df_all['T'].mean()
        acc_list = [acc_train, 0, 0, 0,
                    acc_test, 0, 0, 0]
        print("Accuracy list for attack{} is {}".format(attack_type, acc_list))
        if fair_sample:
            with open(saving_path + '/{}_MIA-acc_fair_attack{}.pkl'.format(dataset, attack_type), 'wb') as f:
                pkl.dump(acc_list, f)
        else:
            with open(saving_path + '/{}_MIA-acc_attack{}.pkl'.format(dataset, attack_type), 'wb') as f:
                pkl.dump(acc_list, f)
        return [attack_type] + acc_list
    for i in range(len(df_all)):
        min_gender, maj_gender = np.unique(gender, return_counts=True)[0][np.unique(gender, return_counts=True)[1].argsort()]
        if df_all.loc[i, 'Node1 Gender'] == min_gender and df_all.loc[i, 'Node2 Gender'] == min_gender:
            df_all.loc[i, 'Group'] = 1
        if df_all.loc[i, 'Node1 Gender'] == maj_gender and df_all.loc[i, 'Node2 Gender'] == maj_gender:
            df_all.loc[i, 'Group'] = 2
    df_all['T'] = df_all['Label'] == df_all['Pred']
    df_all = df_all.set_index('Group')
    acc_test = df_all['T'].mean()
    acc_test0 = df_all.loc[0]['T'].mean()
    acc_test1 = df_all.loc[1]['T'].mean()
    acc_test2 = df_all.loc[2]['T'].mean()
    if attack_type in [1, 5]:
        acc_train = (np.array(y_train_label) == y_train_pred).mean()
        acc_train1 = 0
        acc_train2 = 0
        acc_train0 = 0
    elif attack_type not in [3, 6]:
        tmp_acc = [(np.array(y_train_label) == y_train_pred).mean()]
        for gi in [1, 2, 0]:
            ind_gi_train = g_train == gi
            acc_gi = (np.array(y_train_label)[ind_gi_train] == np.array(y_train_pred)[ind_gi_train]).mean()
            tmp_acc.append(acc_gi)

        acc_train, acc_train1, acc_train2, acc_train0 = tmp_acc
    else:
        train_res = np.array([y_train_label, y_train_pred]).T
        train_res = np.hstack([train_res, id_train])
        gender_pair = []
        for row in train_res:
            node1_gender = int(gender[int(row[2])])
            node2_gender = int(gender[int(row[3])])
            gender_pair.append([node1_gender, node2_gender])

        train_res = np.hstack([train_res, gender_pair])
        df_train = pd.DataFrame(train_res,
                                columns=["Label",
                                         "Pred",
                                         "Node1",
                                         "Node2",
                                         "Node1 Gender",
                                         "Node2 Gender"])
        df_train['Group'] = 0
        for i in range(len(df_train)):
            min_gender, maj_gender = np.unique(gender, return_counts=True)[0][np.unique(gender, return_counts=True)[1].argsort()]
            if df_train.loc[i, 'Node1 Gender'] == min_gender and df_train.loc[i, 'Node2 Gender'] == min_gender:
                df_train.loc[i, 'Group'] = 1
            if df_train.loc[i, 'Node1 Gender'] == maj_gender and df_train.loc[i, 'Node2 Gender'] == maj_gender:
                df_train.loc[i, 'Group'] = 2
        df_train['T'] = df_train['Label'] == df_train['Pred']
        df_train = df_train.set_index('Group')
        acc_train = df_train['T'].mean()
        acc_train0 = df_train.loc[0]['T'].mean()
        acc_train1 = df_train.loc[1]['T'].mean()
        acc_train2 = df_train.loc[2]['T'].mean()
    acc_list = [acc_train, acc_train1, acc_train2, acc_train0,
                acc_test, acc_test1, acc_test2, acc_test0]
    print("Accuracy list for attack{} is {}".format(attack_type, acc_list))
    if fair_sample:
        with open(saving_path + '/{}_MIAacc_fair_attack{}.pkl'.format(dataset, attack_type), 'wb') as f:
            pkl.dump(acc_list, f)
    else:
        with open(saving_path + '/{}_MIAacc_attack{}.pkl'.format(dataset, attack_type), 'wb') as f:
            pkl.dump(acc_list, f)
    return [attack_type] + acc_list


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        features = sp.csr_matrix(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return torch.FloatTensor(adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).todense())


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if isinstance(adj, torch.Tensor):
        if adj.is_sparse:
            adj = adj.to_dense()
        adj = sp.csr_matrix(adj)
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update(
        {placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(
        adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_rm_adj(dataset, rm, adj_ori):
    adj = copy.deepcopy(adj_ori)
    if len(rm) > 0:
        if '[' not in rm:
            #assert rm in ['similarity', 'degree', 'ebc', 'random']
            file = open(f'data/{dataset}/ind.{dataset}-{rm}.adj', 'rb')
            adj = pkl.load(file)
            file.close()
        else:
            n1, n2 = int(rm.split(' ')[0].split('[')[1]), int(rm.split(' ')[1].split(']')[0])
            adj[n1, n2] = 0
            adj[n2, n1] = 0
    return adj


def prepare_testing_adj(adj, non_edges=[], ne_ratio=1.0):
    edges = torch.nonzero(adj).detach().numpy()
    edges = edges[edges[:, 0] < edges[:, 1]]
    if len(non_edges) == 0:
        non_edges = torch.nonzero(1 - adj.fill_diagonal_(1)).detach().numpy()
        non_edges = non_edges[non_edges[:, 0] < non_edges[:, 1]]
    non_edges = non_edges[np.random.choice(len(non_edges), int(ne_ratio * len(edges)))]
    return edges, non_edges


def get_inter_edges(edge1, edge2):
    edge1 = set([tuple(sorted(e)) for e in edge1])
    edge2 = set([tuple(sorted(e)) for e in edge2])
    return np.array(edge1.intersection(edge2))


def get_inter_inds(edge1, edge2, tf_labels):
    assert len(edge1) <= len(edge2)
    s1 = set([tuple(e) for e in edge1])
    s2 = set([tuple(e) for e in edge2])
    return np.array([tf_labels[0] if tuple(edge2[i]) in s1.intersection(s2) else tf_labels[1] for i in range(len(edge2))])


def remove_training(model, dataset, rm, num_sub):
    if torch.cuda.is_available() and os.name == 'nt':
        saving_path = model
    else:
        saving_path = f"../../../../Wang-ds/dzhong2/adversarial-attack/{model}"
    if len(rm) == 0:
        add_on = ""
    else:
        add_on = f"-{rm}"
    for i in range(num_sub):
        for t in range(10):
            try:
                os.remove(f"{saving_path}/{dataset}/subgraph-{i}/mia-input-{t}{add_on}.csv")
            except OSError:
                pass



if __name__ == '__main__':

    datapath_str = "data/dataset/original/"
    dataset_str = "citeseer"
    load_data(datapath_str, dataset_str)

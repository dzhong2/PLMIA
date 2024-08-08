import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from sklearn import metrics
from sklearn.preprocessing import StandardScaler


class MLP3Layer(nn.Module):
    def __init__(self, in_dim, out_dim, layer_list=[64, 64, 32]):
        super(MLP3Layer, self).__init__()
        assert len(layer_list) == 3

        self.fc1 = nn.Linear(in_dim, layer_list[0])
        self.fc2 = nn.Linear(layer_list[0], layer_list[1])
        self.fc3 = nn.Linear(layer_list[1], layer_list[2])
        self.fc4 = nn.Linear(layer_list[2], out_dim)
        self.device = torch.device('cpu')
        self.outdim = out_dim
        self.indim = in_dim
        self.criterion = None
        self.optimizer = None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

    def train_one_epoch(self, Xtrain, ytrain):
        self.train()
        self.optimizer.zero_grad()
        outputs = self(torch.Tensor(Xtrain).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytrain).to(self.device))
        loss.backward()
        self.optimizer.step()

    def loss_acc(self, Xtest, ytest):
        self.eval()
        outputs = self(torch.Tensor(Xtest).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytest).to(self.device))
        acc = (outputs.argmax(dim=1) == torch.LongTensor(ytest).to(self.device)).sum() / len(outputs)

        return loss.cpu().detach().item(), acc.cpu().detach().item()

    def all_metrics(self, X_target, y_target, verbos=True):
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()

        acc_target = metrics.accuracy_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        prec_target = metrics.precision_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        recall_target = metrics.recall_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        auc_target = metrics.roc_auc_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        f1_target = metrics.f1_score(y_target, outputs_target.detach().numpy().argmax(axis=1))

        tpr_05fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.05)
        tpr_10fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.10)
        tpr_01fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.01)

        if verbos:
            print("Accuracy = {:.2%}\n Precision = {:.2%} \n Recall = {:.2%}\n AUC = {:.4}\n F1={:.4}\n TPR@ 1%FPR={:.4}\n TPR@ 5%FPR={:.4}\n TPR@ 10%FPR={:.4} ".format(acc_target,
                                                                                                             prec_target,
                                                                                                             recall_target,
                                                                                                             auc_target,
                                                                                                             f1_target,
                                                                                                                                                                         tpr_01fpr,
                                                                                                                                                                         tpr_05fpr,
                                                                                                                                                                         tpr_10fpr))
        return [acc_target, prec_target, recall_target, auc_target, f1_target, tpr_01fpr, tpr_05fpr, tpr_10fpr]

def tpr_at_fpr(y_true, y_score, fpr_th):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    ind = np.argmin(np.abs(fpr - fpr_th))
    tpr_res = tpr[ind]
    return tpr_res


class MLP2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, layer_list=[64, 32], device=torch.device('cpu')):
        super(MLP2Layer, self).__init__()
        assert len(layer_list) == 2

        self.fc1 = nn.Linear(in_dim, layer_list[0])
        self.fc2 = nn.Linear(layer_list[0], layer_list[1])
        self.fc3 = nn.Linear(layer_list[1], out_dim)

        self.outdim = out_dim
        self.indim = in_dim

        self.device = torch.device('cpu')
        self.criterion = None
        self.optimizer = None
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    def train_one_epoch(self, Xtrain, ytrain):
        self.train()
        self.optimizer.zero_grad()
        outputs = self(torch.Tensor(Xtrain).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytrain).to(self.device))
        loss.backward()
        self.optimizer.step()

    def loss_acc(self, Xtest, ytest):
        self.eval()
        outputs = self(torch.Tensor(Xtest).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytest).to(self.device))
        acc = (outputs.argmax(dim=1) == torch.LongTensor(ytest).to(self.device)).sum() / len(outputs)

        return loss.cpu().detach().item(), acc.cpu().detach().item()

    def all_metrics(self, X_target, y_target, verbos=True):
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()

        acc_target = metrics.accuracy_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        prec_target = metrics.precision_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        recall_target = metrics.recall_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        auc_target = metrics.roc_auc_score(y_target, outputs_target.detach().numpy()[:, 1])
        f1_target = metrics.f1_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        tpr_05fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.05)
        tpr_10fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.10)
        tpr_01fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.01)

        if verbos:
            print("Accuracy = {:.2%}\n Precision = {:.2%} \n Recall = {:.2%}\n AUC = {:.4}\n F1={:.4}\n TPR@ 1%FPR={:.4}\n TPR@ 5%FPR={:.4}\n TPR@ 10%FPR={:.4} ".format(acc_target,
                                                                                                                                                                         prec_target,
                                                                                                                                                                         recall_target,
                                                                                                                                                                         auc_target,
                                                                                                                                                                         f1_target,
                                                                                                                                                                         tpr_01fpr,
                                                                                                                                                                         tpr_05fpr,
                                                                                                                                                                         tpr_10fpr))
        return [acc_target, prec_target, recall_target, auc_target, f1_target, tpr_01fpr, tpr_05fpr, tpr_10fpr]

    def pred_proba(self, X):
        outputs_target = self(torch.Tensor(X).to(self.device)).cpu()
        return outputs_target.detach().numpy()

    def pred(self, X):
        outputs_target = self(torch.Tensor(X).to(self.device)).cpu()
        return outputs_target.detach().numpy().argmax(axis=1)


class MLP1Layer(nn.Module):
    def __init__(self, in_dim, out_dim, layer_list=[64, 32], device=torch.device('cpu')):
        super(MLP1Layer, self).__init__()
        assert len(layer_list) == 1

        self.fc1 = nn.Linear(in_dim, layer_list[0])
        self.fc2 = nn.Linear(layer_list[0], out_dim)

        self.outdim = out_dim
        self.indim = in_dim

        self.device = torch.device('cpu')
        self.criterion = None
        self.optimizer = None
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

    def train_one_epoch(self, Xtrain, ytrain):
        self.train()
        self.optimizer.zero_grad()
        outputs = self(torch.Tensor(Xtrain).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytrain).to(self.device))
        loss.backward()
        self.optimizer.step()

    def loss_acc(self, Xtest, ytest):
        self.eval()
        outputs = self(torch.Tensor(Xtest).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytest).to(self.device))
        acc = (outputs.argmax(dim=1) == torch.LongTensor(ytest).to(self.device)).sum() / len(outputs)

        return loss.cpu().detach().item(), acc.cpu().detach().item()

    def all_metrics(self, X_target, y_target, verbos=True):
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()

        acc_target = metrics.accuracy_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        prec_target = metrics.precision_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        recall_target = metrics.recall_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        auc_target = metrics.roc_auc_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        f1_target = metrics.f1_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        if verbos:
            print("Accuracy = {:.2%}\n Precision = {:.2%} \n Recall = {:.2%}\n AUC = {:.4}\n F1{:.4}".format(acc_target,
                                                                                                             prec_target,
                                                                                                             recall_target,
                                                                                                             auc_target,
                                                                                                             f1_target))
        return [acc_target, prec_target, recall_target, auc_target, f1_target]


def train_model(model, train_x, train_y, test_x, test_y, max_patient=20, display=-1):
    pbar = tqdm(range(200), leave=False, desc=f"Attack {display}" if display != -1 else "")

    opt_loss = 1e10
    patient = max_patient
    for i in pbar:
        model.train_one_epoch(train_x, train_y)

        train_loss, train_acc = model.loss_acc(train_x, train_y)
        val_loss, val_acc = model.loss_acc(test_x, test_y)  # todo: validation

        pbar.set_postfix({'Loss': train_loss,
                          'Acc': train_acc,
                          'Val Loss': val_loss,
                          'Val Acc': val_acc})
        if opt_loss / 1.001 > val_loss:
            opt_loss = val_loss
            patient = max_patient
        else:
            patient = patient - 1

        if patient == 0:
            pbar.close()
            #print("Early break at epoch {}".format(i))

            break
    #print("Training End")
    return model
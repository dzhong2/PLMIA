from models import MLP2Layer, train_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch


def attack_step(model, dataset, rm, num_sub=20, device=torch.device('cuda:0'), number_target=10):

    if len(rm) > 0:
        to_add = "-" + rm
    else:
        to_add = ""
    for i in range(num_sub):
        for t in range(number_target):
            mia_input = pd.read_csv(f"{model}/{dataset}/subgraph-{i}/mia-input-{t}{to_add}.csv", header=None)

            # ss = StandardScaler()
            mia_input = np.array(mia_input)
            # mia_input[:, 3:] = ss.fit_transform(mia_input[:, 3:])
            df_train, df_test = train_test_split(np.array(mia_input), test_size=0.5)
            mia_ft_train = df_train[:, 3:]
            mia_label_train = df_train[:, 2]
            ss = StandardScaler()
            mia_ft_train = ss.fit_transform(mia_ft_train)

            mia_ft_test = df_test[:, 3:]
            mia_label_test = df_test[:, 2]
            mia_ft_test = ss.fit_transform(mia_ft_test)

            attack_model = MLP2Layer(in_dim=mia_ft_train.shape[1],
                                     out_dim=2,
                                     layer_list=[32, 32],
                                     device=device
                                     )
            attack_model.criterion = torch.nn.CrossEntropyLoss()
            attack_model.optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001, weight_decay=1e-5)
            attack_model.to(device)
            attack_model = train_model(attack_model,
                                       mia_ft_train, mia_label_train,
                                       mia_ft_test, mia_label_test,
                                       max_patient=50,
                                       display=i)
            performance = attack_model.all_metrics(mia_ft_test, mia_label_test, verbos=False)
            if performance[0]< 0.6:
                print("Check!")
            torch.save(attack_model.state_dict(), f"{model}/{dataset}/subgraph-{i}/attack_model-{t}{to_add}.dict")
            '''
            train_label = np.ones([len(mia_ft_train), 1])
            pred_mem = attack_model(torch.FloatTensor(mia_ft_train).to(device)).cpu().detach().numpy()
            label_mem = mia_label_train.reshape(-1, 1)

            test_label = np.zeros([len(mia_ft_test), 1])
            pred_nme = attack_model(torch.FloatTensor(mia_ft_test).to(device)).cpu().detach().numpy()
            label_nme = mia_label_test.reshape(-1, 1)

            attack_res = np.vstack([np.hstack([df_train[:, :2],
                                               train_label,
                                               pred_mem,
                                               label_mem]),
                                    np.hstack([df_test[:, :2],
                                               test_label,
                                               pred_nme,
                                               label_nme])
                                    ])

            df_attack = pd.DataFrame(attack_res,
                                     columns=["node1", "node2", "Train-test", "Pred_nm", "Pred_m", "Member"])
            df_attack.to_csv(f"{model}/{dataset}/subgraph-{i}/attack-output-{t}{to_add}.csv", index=False)'''
            del attack_model


if __name__ == "__main__":
    for dataset in ['pubmed']:
        for model in ["GCN"]:
            print(f"running attack on {model} - {dataset}")
            #for rm in ['similarity', 'ebc', 'degree']:
            for rm in ['']:
                attack_step(model, dataset, rm, num_sub=20)

# PLMIA
 
This repository is for paper "Enhancing Link Membership Inference Attacks against Graph Neural Networks through Graph Poisoning", which is submitted to IJCAI 2025

## Run proposed attack
### attack feature generation

`python New_prepare.py --dataset citeseer --partial 0.15 --budget 0.01`

`--dataset`: dataset in {citeseer, pokec, AIDS}

`--partial`: the ratio of training edges to be adversarial graph

`--budget`: poisoning ratio

### attack model training and inference

`python PLMIA_attack.py --dataset citeseer --partial 0.15 --budget 0.01`

`--dataset`: dataset in {citeseer, pokec, AIDS}

`--partial`: the ratio of training edges to be adversarial graph

`--budget`: poisoning ratio

## Run proposed defense

#### Target experiment: Attack multiple targets:

`python PLMIA_attack.py --dataset citeseer --partial 0.15 --budget 0.01 --exp2`

Adding `exp2` will attack multiple targets in one attack, instead of launch one attack for each target.

#### Attack training

`python BatchPoison.py --dataset citeseer --partial 0.15 --budget 0.01`

Train attack model

#### Defense experiment

Apply ParNoise, the defense method to defend against the attack

`--beta`: Noise scale of laplace noise. A larger beta indicates a stronger defense.

## implement of GNN models
We implement the GCN model based on [this repo](https://github.com/tkipf/pygcn). The codes are in the `pygcn` folder.
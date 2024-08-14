# MEDUSA
 
This repository is for paper "Exploiting Graph Poisoning and Unlearning to Enhance Link Inference Attacks", which is submitted to AAAI 2025

## Run proposed attack
### attack feature generation

`python New_unlearnLeak.py --dataset citeseer --partial 0.15 --budget 0.01`

`--dataset`: dataset in {cora, citeseer, pubmed}

`--partial`: the ratio of training edges to be adversarial edge

`--budget`: poisoning ratio

### attack model training and inference

`python UnlearnLeakAttack.py --dataset citeseer --partial 0.15 --budget 0.01`

`--dataset`: dataset in {cora, citeseer, pubmed}

`--partial`: the ratio of training edges to be adversarial edge

`--budget`: poisoning ratio

## Run proposed defense

#### Target experiment: Attack multiple targets:

`python UnlearnLeakAttack.py --dataset citeseer --partial 0.15 --budget 0.01 --exp2`

Adding `exp2` will attack multiple targets in one attack, instead of launch one attack for each target.

#### Attack training

`python BatchUnlearnLeak.py --dataset citeseer --partial 0.15 --budget 0.01`

Train attack model

#### Defense experiment

Apply DEU, the defense method to defend against the attack

`--beta`: Noise scale of laplace noise. A larger beta indicates a stronger defense.

## implement of GNN models
We implement the GCN model based on the [this repo](https://github.com/tkipf/pygcn). The codes are* *in the `pygcn` folder.
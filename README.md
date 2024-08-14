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


## implement of GNN models
We implement the GCN model based on the [this repo](https://github.com/tkipf/pygcn). The codes are in the `pygcn` folder.
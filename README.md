# How to use this repository

## 1. original data preparation

Download tacred dataset, and put train.json, dev.json, test.json into data/tacred

Download tacrev's dev_rev.json, test_rev.json, and put them into data/tacred

Download retacred dataset, and put train.json, dev.json, test.json into data/retacred

The entity-dict-wiki.json is collected by us from wikidata, and it will be used to generate the challenge data.

The data directory should be like this:

```tree
data
├── entity-dict-wiki.json
├── retacred
│   ├── dev.json
│   ├── test.json
│   └── train.json
└── tacred
    ├── dev.json
    ├── dev_rev.json
    ├── test.json
    ├── test_rev.json
    └── train.json
```

The output of all models will be saved into the results directory.

## 2. Environment setup

create the python virtual environment by executing the scripts/install_env.sh (This file is just a wrapper of the
environment installation script of LUKE for simplify usage)

environment name should be any valid name for the conda environment, we will refer to it as <env_name> in the following

```bash
bash install_env.sh <env_name>
```

## 3. Generate training data

```bash
bash preprocess/gen_train_files.sh
```

## 4. Generate challenge data for OOD evaluation

### 4.1 ours challenge data generation

#### 4.1.1 prepare the model

```bash
conda activate <env_name>
cd RE_improved_baseline
bash scripts/train.sh 0 tacred EntityOnly
bash scripts/train.sh 1 retacred EntityOnly
```

#### 4.1.2 generate the challenge data

```bash
bash preprocess/gen_challenge_files.sh 0 tacred 64
bash preprocess/gen_challenge_files.sh 1 retacred 64
```

### 4.2 CoRE challenge data generation

> The CoRE challenge data generation is based on the entity-only setting. We first train the model under the default setting, and then generate the counterfactual samples based on the model's predictions under the entity-only setting.

#### 4.2.1 Train model under the default setting

```bash
# for IBRE
conda activate <env_name>
cd RE_improved_baseline
bash scripts/train.sh 0 tacred default
bash scripts/train.sh 1 retacred default
```

```bash
# for LUKE
conda activate <env_name>
cd luke
bash scripts/train.sh 0 tacred default
bash scripts/train.sh 1 retacred default
```

#### 4.2.2 Generate entity-only predictions

```bash
# for IBRE
conda activate <env_name>
cd RE_improved_baseline
bash scripts/gen_eo_prob.sh 0 tacred default
bash scripts/gen_eo_prob.sh 0 retacred default
```

```bash
# for LUKE
conda activate <env_name>
cd luke
bash scripts/gen_eo_prob.sh 0 tacred default
bash scripts/gen_eo_prob.sh 0 retacred default
```

#### 4.2.3 Generate counterfactual samples based on the entity-only predictions

```bash
bash preprocess/gen_counterfactual.sh tacred
bash preprocess/gen_counterfactual.sh retacred
```

## 5. Debiasing methods

Our mix-debias method support IBRE(RE_improved_baseline) and LUKE. The debiasing methods are implemented in
the `RE_improved_baseline` and `luke` directories. The debiasing methods include:

* default: the original model
* EntityMask: mask the entity mentions in the input
* DataAug: data augmentation by replacing entity mentions with other entities
* RDrop: regularize the model by feedforward twice
* Focal: focal loss
* DFocal: Debiased Focal Loss
* PoE: Product-of-Expert used to integrate the predictions of target model and bias model
* Debias: Casual Debias Approach proposed by us
* RDataAug: Regularized Debias Approach proposed by us
* MixDebias: debiasing method proposed by us

Please refer to the `RE_improved_baseline` and `luke` directories for more details.

## 6. How to clone this repository

This repository contains submodules. To clone this repository, please use the following command:

```bash
git clone --recurse-submodules <repository-url>
```
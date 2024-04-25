## Introduction

This is an re-implementation of the paper [Should We Rely on Entity Mentions for Relation Extraction? Debiasing Relation Extraction with Counterfactual Analysis](https://aclanthology.org/2022.naacl-main.224.pdf) because the original code is not suitable for other models or datasets. This code only requires the output probabilities of the model, so it can be used with any model and dataset.

### Requirements

You need record the output probabilities of the relation classification model base on original dataset and entity-only dataset. besides, you should provide the label2id.json file to map the relation label to id.

use the following command to link the output of the model to this code.
```
bash scripts/setup.sh
```

### Running
```bash
bash eval.sh tacred ibre default 42 dev test
```
The above command means that the model is trained on the TACRED dataset, and the model name is ibre(a Improved Baseline for Relation Extraction), the training mode is default, the random seed is 42, and the dev dataset is  used to select superparameters, and the test dataset is used to evaluate the model.

## License
MIT
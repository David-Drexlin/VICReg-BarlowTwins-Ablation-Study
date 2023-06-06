## Project Description

Tutorial/API ----> :  https://px39n.gitbook.io/autossl/

autoSSL is a versatile and accessible library for Self-Supervised Learning (SSL). This Python-based deep learning library provides you with utilities for data loading, model architectures, evaluation scripts, and numerous experiment notebooks. The design and organization of this project encourage flexible, scalable, and reusable workflows for your SSL tasks.

## Code Structure
```
.
├── autoSSL                      # Main project folder
│   ├── data                     # Data loading utilities
│   │   ├── PipeDataset.py       # Dataset loading pipeline
│   │   └── __init__.py
│   ├── evaluate                 # Evaluation scripts
│   │   ├── eval_KNN.py          # K-Nearest Neighbors evaluation
│   │   ├── eval_KNNplot.py      # K-Nearest Neighbors evaluation plot
│   │   ├── eval_linear.py       # Linear evaluation
│   │   ├── pipe_collate.py      # Pipeline for model collation
│   │   └── __init__.py
│   ├── models                   # SSL model architectures
│   │   ├── Backbone.py          # Backbone for SSL models
│   │   ├── BarlowTwins.py       # Barlow Twins architecture
│   │   ├── BYOL.py              # Bootstrap Your Own Latent (BYOL) architecture
│   │   ├── MoCo.py              # Momentum Contrast (MoCo) architecture
│   │   ├── pipe_model.py        # Model loading pipeline
│   │   ├── SimCLR.py            # SimCLR architecture
│   │   ├── SimSiam.py           # SimSiam architecture
│   │   ├── VICReg.py            # VICReg architecture
│   │   └── __init__.py
│   ├── train                    # Training scripts
│   │   ├── pipe_train.py        # Training pipeline
│   │   └── __init__.py
│   └── utils                    # Utility scripts
│       ├── dict2transformer.py  # Function to transform dictionaries
│       ├── embedding.py         # Embedding functions
│       ├── join_dir.py          # Directory joining function
│       ├── logging.py           # Logging functionalities
│       └── __init__.py
├── experiment                   # Experiment notebooks
│   ├── Experiment1.ipynb
│   ├── Experiment2.ipynb
│   ├── experiment_checkpoints   # Checkpoints for experiments
│   │   └── Experiment1
│   │       ├── barlow_batch_1024
│   │       │   ├── barlow_batch_1024
│   │       │   │   └── version_0
│   │       │   │       └── hparams.yaml
│   │       │   └── config.yaml
│   │       ├── barlow_batch_128
│   │       │   ├── barlow_batch_128
│   │       │   │   └── version_0
│   │       │   │       └── hparams.yaml
│   │       │   └── config.yaml
│   │       └── batch_[0-9]+.csv
│   ├── global.yaml              # Global configurations
│   └── Unit_Test.ipynb          # Unit tests
├── experiment_template.ipynb   # Template for creating new experiments
├── README.md                    # This file
├── LICENSE                      # License file
└── Template_Legacy.ipynb        # Legacy template
```

## Quick Start Guide

### Training

First, set up your own configuration. This example demonstrates a configuration for the BarlowTwins model with a batch size of 512.

```python
config = global_config.copy()
SSL_augmentation = global_SSL_augmentation
config["name"] = "barlow_batch_512"
config["batch"] = 512
config["model"] = "BarlowTwins"

# Load the dataset
pdata = PipeDataset(config=config, augmentation=dict2transformer(SSL_augmentation, view=config["view"]))

# Load the model
pmodel = pipe_model(config=config)

# Initialize a trainer for the model and begin training
trainer = Trainer(config, model_mode="start")
trainer.fit(pmodel, pdata.dataloader)
# cONTINUE to train
#trainer1 = Trainer(config, model_mode="continue", extra_epoch=0)
#trainer1.fit(pmodel, pdata.dataloader, ckpt_path="latest")
```

### Evaluation

Perform evaluation using the trained models:

```python
collate = pipe_collate(address="experiment_checkpoints/batch VS model/", reg="batch_[0-9]+")
aaa = eval_linear(pdata, models=collate, device=global_config["device"], split=0.8)
```

The `autoSSL.evaluate` module offers a suite of functions for model evaluation, including `eval_KNN` for K-Nearest Neighbors evaluation, `eval_KNNplot` for visualizing K-Nearest Neighbors results, and `eval_linear` for linear evaluation of the model.

Please explore the project and refer to the individual experiment notebooks for more detailed examples and explanations. We look forward to seeing what you will create with autoSSL!


### Configuration Summary

 
1. **Project Configuration**

| Hyperparameter | Description                   | Example Values                                   | Used in Function            |
| -------------- | ----------------------------- | ------------------------------------------------ | --------------------------- |
| checkpoint_dir | Directory to save checkpoints | 'experiment_checkpoints/'                        | ck_callback(checkpoint_dir) |
| experiment     | Name of the experiment        | "batch VS model"                                 | For pipe Used                        |
| name           | Specific configuration name   | "config1"                                        | pipe_model(name=...)        |
| log_dir        | Directory to save logs        | 'experiment_checkpoints/batch VS model/config1/' | For pipe Used                        |

2. **Model Configuration**

| Hyperparameter | Description | Example Values | Used in Function |
| -------------- | ----------- | -------------- | ---------------- |
| model | The SSL model to use | "VICReg", "MoCo", "BYOL", "SimCLR", "SimSiam", "BarlowTwins" | pipe_model(name=...) |
| backbone | The backbone model for SSL model | "resnet18", "resnet50", "efficientnet_b5", "mobilenet_v3", "vit_64", "vit_224" | pipe_model(backbone=...) |
| stop_gradient | Whether to stop gradient or not | True, False | pipe_model(stop_gradient=...) |
| prjhead_dim | The dimension of projection head | 2048, 1024, 512 | pipe_model(prjhead_dim=...) |

3. **Training Configuration**

| Hyperparameter | Description | Example Values | Used in Function |
| -------------- | ----------- | -------------- | ---------------- |
| max_epochs | The maximum number of epochs for training | 5, 10, 50 | Trainer(config, model_mode, extra_epoch=...) |
| device | The device for training | "cuda", "cpu" | eval_linear(device=...), eval_KNN(device=...), eval_KNNplot(device=...) |

4. **Dataset Configuration**

| Hyperparameter | Description | Example Values | Used in Function |
| -------------- | ----------- | -------------- | ---------------- |
| input_size | The input size for the model | 64, 128, 224 | for augmentation |
| path_to_train_cifar10 | The path to the training dataset cifar10 | "../../Datasets/cifar10/train/" | PipeDataset(input_dir=...) |
| path_to_test_cifar10 | The path to the testing dataset cifar10 | "../../Datasets/cifar10/test/" | PipeDataset(input_dir=...) |
| view | The number of views for each instance | 1, 2 | dict2transformer(dict, view=...) |
| samples | The number of samples to load from the dataset | 0, 100, 1000 | PipeDataset(samples=...) |
| batch_size | The size of each batch during training | 512, 256, 128 | PipeDataset(batch_size=...) |
| shuffle | Whether to shuffle the dataset | True, False | PipeDataset(shuffle=...) |
| drop_last | Whether to drop the last incomplete batch during training | True, False | PipeDataset(drop_last=...) |
| num_workers | The number of worker threads for data loading | 4, 8, 16 | PipeDataset(num_workers=...) |


## More
 

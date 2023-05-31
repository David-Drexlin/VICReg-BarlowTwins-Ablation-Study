# Project Description

This project contains a Python-based deep learning library called `autoSSL`, designed for Self-Supervised Learning (SSL) tasks. It includes utilities for data loading, several model architectures, evaluation scripts, as well as various experiment notebooks. The project is organized into a structure that supports a flexible, scalable, and reusable workflow for SSL tasks.

## Code Structure

```
.
├── autoSSL
│   ├── utils
│   │   └── dataloader2array.py
│   │   └── ...
│   ├── data
│   │   └── pipe_dataloader.py
│   │   └── ...
│   ├── models
│   │   ├── BYOL.py
│   │   └── barlowtwins.py
│   │   └── ...
│   └── evaluate
│       ├── eval_knn.py
│       └── eval_linear.py
│   │   └── ...
├── checkpoints
│   ├── experiment1
│   ├── experiment2
│   ├── ...
├── experiments
│   ├── experiment1.ipynb
│   ├── experiment1.yaml 
│   ├── experiment2.ipynb
│   ├── experiment2.yaml 
│   ├── ...
├── global.yaml
├── readme.md
└── ...


```
The main components of the project are contained within the `autoSSL` directory, and include:

- `utils`: Utility scripts, including `dataloader2array.py` and more, for data manipulation and processing.
- `models`: Scripts defining several model architectures including `BYOL.py`, `barlowtwins.py` and others.
- `evaluate`: Scripts for model evaluation such as `eval_knn.py`, `eval_linear.py`, and more.
- `pipe_dataloader`: Augmentation etc,.
There are also a number of Jupyter notebooks contained within the `experiments` directory, each one documenting a different experiment performed with the `autoSSL` library. Associated with each experiment notebook is a corresponding YAML configuration file, allowing for localized adjustments of parameters.

Checkpoints from each experiment, including different model states and epochs, are stored in the `checkpoints` directory. Each checkpoint file corresponds to a specific experiment.


# Quick API



| Model Name    | Code                                                         | Description                                                  |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| BarlowTwins   | `model = BarlowTwins(backbone='ResNet', memory_bank=False, projectionhead='linear')` | Barlow Twins is a model that inherits from the `General_Model` class and utilizes the lightly model. It uses a ResNet backbone, does not employ a memory bank, and has a linear projection head. |
| MoCo          | `model = MoCo(backbone='ResNet', memory_bank=True, projectionhead='MLP')` | MoCo (Momentum Contrast) is a model that inherits from the `General_Model` class and utilizes the lightly model. It uses a ResNet backbone, employs a memory bank, and has an MLP projection head. |
| SimCLR        | `model = SimCLR(backbone='ResNet', memory_bank=False, projectionhead='linear')` | SimCLR (Simple Contrastive Learning) is a model that inherits from the `General_Model` class and utilizes the lightly model. It uses a ResNet backbone, does not employ a memory bank, and has a linear projection head. |
| SiaSiam       | `model = SiaSiam(backbone='ResNet', memory_bank=True, projectionhead='MLP')` | SiaSiam is a model that inherits from the `General_Model` class and utilizes the lightly model. It uses a ResNet backbone, employs a memory bank, and has an MLP projection head. |
| BYOL          | `model = BYOL(backbone='ResNet', memory_bank=False, projectionhead='linear')` | BYOL (Bootstrap Your Own Latent) is a model that inherits from the `General_Model` class and utilizes the lightly model. It uses a ResNet backbone, does not employ a memory bank, and has a linear projection head. |




# Optimizing

Train


```python
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
   monitor='train_loss',
   dirpath='../VICReg-BarlowTwins-Ablation-Study/archi_weights/VICReg',
   filename='cifar10-{epoch:02d}-{train_loss:.2f}',
   save_top_k=3,
   mode='min',
)

trainer = pl.Trainer(max_epochs=max_epochs, devices=1, accelerator="gpu", callbacks=[checkpoint_callback])
trainer.fit(model, dataloader_train)
```
 Load the Model
```python
import pytorch_lightning as pl

model =VICReg()

trainer = pl.Trainer(max_epochs=2, devices=1, accelerator="gpu", callbacks=[checkpoint_callback])
trainer.fit(model, dataloader_train, dataloader_test,ckpt_path='../.ckpt')
```

# Evaluation

| Function         | Description                                    | Parameters                                                                     | Returns                                                                  | Example Code                                         |
|------------------|------------------------------------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------|------------------------------------------------------|
| `Eval_transfer`  | Evaluates transfer learning performance        | `model`, `target_dataset`, `metric`, `top_k`, `average`                        | Evaluation score(s) based on the specified metric(s)                     | `Eval_transfer(model=pretrained_model, target_dataset=(X, y), metric='accuracy')` |
| `Eval_linear`    | Evaluates linear model performance             | `model`, `dataset`, `metric`, `top_k`, `average`                                 | Evaluation score(s) based on the specified metric(s)                     | `Eval_linear(model=linear_model, dataset=(X, y), metric='accuracy')`             |
| `Eval_knn`       | Evaluates KNN model performance                | `model`, `dataset`, `metric`, `top_k`, `average`                                 | Evaluation score(s) based on the specified metric(s)                     | `Eval_knn(model=knn_model, dataset=(X, y), metric='accuracy')`                    |
| `Compare_eval`   | Compares the performance of multiple models    | `models`, `dataset`, `metrics`, `top_k`, `average`                               | Evaluation scores for each model and metric combination                  | `Compare_eval(models=[logistic_regression, knn], dataset=(X, y), metrics=['accuracy', 'precision', 'recall'])` |

The table above provides a summary of the functions in the `autoSSL.evaluate` module. It includes the function name, a short description of its purpose, the parameters it accepts, the returns it provides, and an example code snippet demonstrating the function's usage.

# For example

If we check multi modal

If we check same model in differen dataset

If we check different backbone

If we????
Etc,.

hi there, after the meeting with Patrik, the following research topic will be our main focus:

* impact of the nr. of distorted samples on the accuracy/loss
* focus on one computationally feasible architecture (VICReg, Barlow-Twins)
* objective function:
    * accuracy: replacement of the decoder (projector) with a linear classifier → supervised training of the linear classifier necessary
    * loss: keep architecture
* preliminary study: pretraining resnet weights vs. random initializing weights


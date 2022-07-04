# Introduction

**SMRB** is a benchmark for service mashup recommendation to solve the problem of irregularities in the field of service mashup recommendation. 

> This project is built on open source project [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

## motivation

Currently, deep learning based approaches in service mashup recommendation have common problems, including the non-unified dataset pre-trained model, evaluation rotocol, andexperiment environment. These issues will disrupt evaluating the performance of models accurately and make reproducing them difficult. **SMRB** rovides a standard environment to enhance comparability between models and credibility of results.

# Project Structure

The directory structure of new project looks like this:

```
│   .env.example                   <- Template of the file for storing private environment variables
│   .gitignore                     <- List of files/folders ignored by git
│   .pre-commit-config.yaml        <- Configuration of pre-commit hooks for code formatting
│   README.md
│   requirements.txt               <- File for installing python dependencies
│   setup.cfg                      <- Configuration of linters and pytest
│   test.py                        <- Run testing
│   train.py                       <- Run training
│
├───configs                        <- Hydra configuration files
│   │   test.yaml                     <- Main config for testing
│   │   train.yaml                    <- Main config for training
│   │
│   ├───callbacks                  <- Lightning callbacks
│   │       wandb.yaml                <- Wandb and metrics callbacks
│   │
│   ├───datamodule                    
│   │       partial_text_bert.yaml    <- Partial text-based dataset embedded by BERT configs
│   │       partial_text_glove.yaml   <- Partial text-based dataset embedded by GloVe configs
│   │       partial_word_bert.yaml    <- Partial word-based dataset embedded by BERT configs
│   │       partial_word_glove.yaml   <- Partial word-based dataset embedded by GloVe configs
│   │       total_text_bert.yaml      <- Total text-based dataset embedded by BERT configs
│   │       total_text_glove.yaml     <- Total text-based dataset embedded by GloVe configs
│   │       total_word_bert.yaml      <- Total word-based dataset embedded by BERT configs
│   │       total_word_glove.yaml     <- Total word-based dataset embedded by GloVe configs
│   │
│   ├───experiment                 <- Experiment configs
│   │
│   ├───hparams_search             <- Hyperparameter search configs
│   │
│   ├───logger                     <- Logger configs
│   │
│   ├───log_dir                    <- Logging directory configs
│   │
│   ├───model                      <- Model configs
│   │
│   └───trainer                    <- Trainer configs
│
├───data                        <- Project data
│
├───logs                        <- Logs generated by Hydra and PyTorch 
                                   Lightning loggers
├───src                         <- Source code
│   │   testing_pipeline.py
│   │   training_pipeline.py
│   │
│   ├───callbacks
│   │       wandb_callbacks.py
│   │
│   ├───datamodules             <- Lightning datamodules
│   │
│   ├───models                  <- Lightning models
│   │
│   ├───utils                   <- Utility scripts
│   │
│   └───vendor                  <- Third party code that cannot be installed using PIP/Conda
│
└───tests                       <- Tests of any kind
    │
    ├───helpers                    <- A couple of testing utilities
    │
    ├───shell                      <- Shell/command based tests
    │
    └───unit                       <- Unit tests
```

Download the data from [here](https://drive.google.com/file/d/1iOgjbF2Hdk4t9zE4VHRgfYeurD7_PVog/view?usp=sharing) and copy it to the *data* directory.

# Installation
## Install with anaconda
```
# clone project
git clone https://github.com/ssnowyu/SMRB
cd SMRB

# create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install requirements
pip install -r requirements.txt
```

## Install with docker
You will need to install Nvidia Container Toolkit to enable GPU support.

```
# clone project
git clone https://github.com/ssnowyu/SMRB
cd SMRB

# build the container
docker build -t <project_name> .

# mount the project to the container
docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>
```

# Quickstart

1. Implement the DL-based model as the [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) class. Details refer to [Model Implementation](#implement). Here the `MLP` model (pre-configured in our project) is used as an example.
2. Write a configuration file called `simple_model` for your model.
   ```
    _target_: src.models.mlp.MLP

    data_dir: ${data_dir}/api_mashup
    api_embed_path: embeddings/partial/text_bert_api_embeddings.npy
    mashup_embed_channels: 300
    mlp_output_channels: 300
    lr: 0.001
    weight_decay: 0.00001
   ```
3. Write a configuration file called `mlp` for your experiment.
   ```
    # @package _global_

    defaults:
        - override /trainer: default.yaml              # use default settings for trainer
        - override /model: mlp.yaml                    # use "mlp" as model
        - override /datamodule: partial_text_bert.yaml # use partial text-based embeddings encoded by BERT
        - override /callbacks: wandb.yaml              # use wandb as the callbacks
        - override /logger: wandb.yaml                 # use wandb as the log framework

    seed: 12345

    logger:
        wandb:
            name: 'MLP-partial-BERT'
            tags: ['partial', 'BERT']

    # Override model parameters
    model:
        api_embed_path: embeddings/partial/text_bert_api_embeddings.npy
        mashup_embed_channels: 768
        mlp_output_channels: 300
        lr: 0.001
   ```
4. Since the project uses [wandb](https://wandb.ai/site) as the log framework by default, you will need to have a wandb account and bind the account to the project by executing the following command.
   ```
   wandb login
   ```
   This command needs to be executed only once during the entire development process.

   If you do not want to use wandb, you can also choose another log framework. Please refer to [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) for how to change it.
5. run the project.
   ```
   python train.py experiment=mlp/partial_bert
   ```
# Guide

## Choose dataset
We provides 8 datasets, as shown in the following table:

| name               | form       | pre-tained model | amount  |
|--------------------|------------|------------------|---------|
| partial_text_bert  | text-based | BERT             | partial |
| partial_text_glove | text-based | GloVe            | partial |
| partial_word_bert  | word-based | BERT             | partial |
| partial_word_glove | word-based | GloVe            | partial |
| total_text_bert    | text-based | BERT             | total   |
| total_text_glove   | text-based | GloVe            | total   |
| total_word_bert    | word-based | BERT             | total   |
| total_word_glove   | word-based | GloVe            | total   |

### Two forms
- $(72 \times d)$. The original form processed by word embedding model, and each word corresponds to a vector whose size is $(1 \times d)$. This form is suitable for word-based representation.
- $(1 \times d)$. Pooling the representation whose size is $(72 \times d)$ by averaging to a representation whose size is $(1 \times d)$, with which the whole text is represented. This form is suitable for text-based representation.

### Two amount
- **total**: 21495 APIs, including some unused APIs.
- **partial**: 932 APIs, all of which have been used at least once.

### Two pre-trained models
- **BERT**: A global log-bilinear regression model for unsupervised learning of word representations.
- **GloVe**: Bidirectional Encoder Representations from Transformer.

## <span id="implement">Model implementation</span>

![pic](https://github.com/tsdwgfaf/pictures/raw/master/simple_model.png "A model implementation")

You need to implement the DL-based model as the [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) class. What you need to do are:

1. return the loss in `training_step` to guide model training.
2. return the predicted result *preds* and the true result *targets* in `test_step`. Then, the callback mechanism will capture them and calculates the performance on each metric.

## Experiment configuration

Organizing experiments by combining existing elements. Take *simple-model* as an example.
![pic](https://github.com/tsdwgfaf/pictures/raw/master/config.png "A experiment config")

## Pre-configured models
1. [MTFM](https://ieeexplore.ieee.org/document/9492754)
2. [coACN](https://ieeexplore.ieee.org/document/9590360/)
3. [MISR](https://ieeexplore.ieee.org/document/8960409)
4. [FISR](https://arxiv.org/abs/2101.02836)
5. [T2L2](https://link.springer.com/chapter/10.1007/978-3-030-91431-8_20)
6. T2L2-W/O-Propagation: This method is based on a modification of T2L2. We remove the propagation component from T2L2 and obtain T2L2-W/O-Propagation.
7. MLP: MLP with two linear layers.
8. Freq: This method always recommends the top $N$ frequently invoked Web APIs.
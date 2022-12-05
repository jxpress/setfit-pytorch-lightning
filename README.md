# Seifit-PyTorch-Lightning

![main_theme](/documents/main_theme.png)

<br>

# 🤗 About SetFit

The [SetFit](https://github.com/huggingface/setfit) provides the strong method of a few shot learning for text classification.
With SetFit, you can create an AI with an accuracy comparable to GPT3 with as little as a few dozen data points.
You can see official [paper](https://arxiv.org/abs/2209.11055), [blog](https://huggingface.co/blog/setfit), and [code](https://github.com/huggingface/setfit) of SetFit.

If you want to experience SetFit, you can access [here](https://github.com/huggingface/setfit/tree/main/notebooks) and find same example notebooks to run SetFit.

This repository provides code that allows [SetFit](https://github.com/huggingface/setfit) to run in [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to facilitate parameter, experiment management and so on.

This repository is created from [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

<br>

# 🚀  How to use this repository

## step 0: create miniconda GPU environment and operation check

**Create miniconda GPU environment**

```bash
# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv
# install requirements
pip install -r requirements.txt
```

**Operation check**

Enter the following code to execute the sample code (classification of sst2).

```bash
make operation-check
```

or

```bash
python src/train.py ++trainer.fast_dev_run=true
```

<br>

## step 1. Custom LightningDataModule.

Data is managed in LightningDataModule.
In the [the sample code](src/datamodules/setfit_datamodule.py), training data is obtained from the sst2 dataset.

If you are not familiar with PyTorch Lightning, I recommend you to change only self.train_dataset, self.valid_dataset and self.test_dataset in `__init__`.
Parameter of Datamodule is managed in [config file](configs/datamodule/setfit.yaml).

Or if you want to custom more, [README of lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) would offer useful information.

<br>

## step 2. Custom LightningModule.

Parameters that were entered into the [original SetFit trainer and SetFitModel](https://github.com/huggingface/setfit) can be entered into [LightnigngModule](src/models/setfit_module.py). You can manage such parameters in [config file](configs/model/setfit_nn.yaml).

If you want to customize more, see [here](documents/Implemented_strategy.md) to find out how we implemented SetFit in PyTorch Lightning

<br>

## step 3. Custom other option such as callback or logger.

PyTorch Lightning offers useful [callbacks](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html) and [logger](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html) to save a model or metrics and so on.
You can manage what and how callback or logger will be called in [config files](configs).

**⚠Note : if you want to use callbacks of ModelCheckpoint, use [SetFitModelCheckpoint](src/utils/callbacks.py) to save model if model head is consist of sklearn, like sample code**

## step 4. Execute the train

Run

```bash
python src/train.py
```

Or you can [override experimental configtion](https://hydra.cc/docs/advanced/override_grammar/basic/) like below

```bash
python src/train.py trainer.max_epochs=1
```

## step 5. Load the trained model

Since SetFIt model may be configured with sklearn, so please load the model as in [this notebook](notebooks/model_load.ipynb).

# 🐾 others

## Experiment management

For managing your experimentm you can add experimental confition to config file [like this](configs/experiment/example.yaml) and run like below

```bash
python src/train.py experiment=example
```

For more information, [this](https://github.com/ashleve/lightning-hydra-template#experiment-config) might useful for you

<br>

## Hyperparameter optimize

IF you want to excepuce hyperparameter optimization, just  add config file [like this](configs/hparams_search/setfit_optuna.yaml) and run like below

```bash
python src/train.py -m hparams_search=setfit_optuna
```

For more information, [this](https://github.com/ashleve/lightning-hydra-template#hyperparameter-search) might useful for you
<br>

## 😍 Welcome contributions

if you find some error or feel something, feel free to tell me by PR or Issues!!
Opinions of any content are welcome!

## 📝 Appendix

JX PRESS Corporation has created and use the training template code in order to enhance team development capability and development speed.

For more information on JX's training template code, see [How we at JX PRESS Corporation devise for team development of R&D that tends to become a genus](https://tech.jxpress.net/entry/2021/10/27/160154) and [PyTorch Lightning explained by a heavy user](https://tech.jxpress.net/entry/2021/11/17/112214). (Now these blogs are written in Japanese. If you want to see, please translate it into your language. We would like to translate it in English and publish it someday)

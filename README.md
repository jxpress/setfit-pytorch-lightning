# Seifit-PyTorch-Lightning

![main_theme](/documents/main_theme.png)

<br>

# 🤗 About SetFit

The [SetFit](https://github.com/huggingface/setfit) provides the strong method of a few shot learning for text classification.
With SetFit, you can create an AI with an accuracy comparable to GPT3 with as little as a few dozen data points.
You can see official [paper](https://arxiv.org/abs/2209.11055), [blog](https://huggingface.co/blog/setfit), and [code](https://github.com/huggingface/setfit) of Setfit.

If you want to experience SetFit, you can access [here](https://github.com/huggingface/setfit/tree/main/notebooks) and find same example notebooks to run SetFit.

This repository provides code that allows [SetFit](https://github.com/huggingface/setfit) to run in [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to facilitate parameter and experiment management.

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

Just type

```bash
make train
```

to execute training of classification of sst2

<br>

## step 1. Custom datamodule.

Train data is managed in LightningDataModule.
[The sample code](src/datamodules/setfit_datamodule.py) creates training data for `num_samples` of data per class from the sst2 data set.

If you want to modify to your custom training, I recommend you to change only `self.train_dataset`, `self.valid_dataset` and `self.test_dataset` in __init__

If you want to create dataset from DataFrame of pandas, you transfer from dataframe to dataset like below

```python
import pyarrow as pa
from datasets import Dataset

self.train_dataset = Dataset(pa.Table.from_pandas(dataframe[["sentence","label"]]))
```

where, we assume that the dataframe has columns for sentence as input and label as target.

<br>

### step 2. confirm that training is executable with Docker Image.

Vertex AI uses Docker Image for training, so it is necessary to confirm the training on Docker Image.
At that time, you can confirm that by typing below in root directory.

```bash
make train-in-docker
```

Option such as checking operation on GPU can be adjusted in [docker-compose.yaml](/docker-compose.yaml).

<br>

### step 3. Prepare a GCP account.

If you do not have a GCP account, please prepare a GCP account from [here](https://cloud.google.com/docs/get-started).
This repository uses [Vertex AI](https://cloud.google.com/vertex-ai/docs/start) and [Artifact Registry](https://cloud.google.com/artifact-registry). Please activate the respective APIs in GCP.

Next, [create a docker repository](https://cloud.google.com/artifact-registry/docs/repositories/create-repos#overview)) to push Docker Images to the Artifact Registry.

Then [determine the name of the Image](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling).

<br>

### step 4-1. Run a custom job

- Set the name and tag of the Image determined in step 3 in imageUri of [vertex_ai/configs/custom_job/default.yaml](/vertex_ai/configs/custom_job/default.yaml).
- Set region, gcp_project in [vertex_ai/scripts/custom_job/create_job.sh](/vertex_ai/scripts/custom_job/create_job.sh).
- In the root folder, type

```bash
make create-custom-job
```

in the root folder.
Docker build and push will be performed, and the custom job of Vertex AI will be started with the pushed image.
You can check the training status at [CUSTOM JOBS](https://console.cloud.google.com/vertex-ai/training/custom-jobs) in the Vertex AI training section of GCP.

<br>

### step 4-2. Run a hyperparameter tuning job

- Set the name and tag of the Image determined in step 3 in imageUri of  [vertex_ai/configs/hparams_tuning/default.yaml](/vertex_ai/configs/hparams_tuning/default.yaml).
- Set the metrics that you want optimize in [configs/hparams_search/vertex_ai.yaml](/configs/hparams_search/vertex_ai.yaml).
- Set region, gcp_project in [vertex_ai/scripts/hparams_tuning/create_job.sh](/vertex_ai/scripts/hparams_tuning/create_job.sh)
- In the root folder, type

```bash
make create-hparams-tuning-job
```

in the root folder.
Docker build and push will be performed, and the hyperparameter tuning job of Vertex AI will be started with the pushed image.

You can check the training status at [HYPERPARAMETER TUNING JOBS](https://console.cloud.google.com/vertex-ai/training/hyperparameter-tuning-jobs) in the Vertex AI training section of GCP.

<br>

# 🔧　Changes

The following changes have been made in this repository from [train template code](https://github.com/ashleve/lightning-hydra-template).

- Dockerfile
  - For build docker image, I copied and slightly modified from [the Dockerfile in the branch of original repository](https://github.com/ashleve/lightning-hydra-template/tree/dockerfiles)
- docker-compose.yaml
  - To check the operation
- configs/hparams_search/vertex_ai.yaml
  - Used in hyperparameter tuning of Vertex AI
- Makefile
  - Add code related to docker and Vertex AI
- folder and code for Vertex AI
  - configs
    - Add yaml file related to settings.
  - script
    - Add code to execute train job in Vertex AI
- requirements.txt
  - Add package for Vertex AI
- README.md
  - Add README.md. Original README is moved to documents folder
- documents
  - Move the original README.md
  - Add the Japanese version of README.md
  - translated blog
    - English translation of a detailed blog about Hydra and Vertex AI.

# 📝 Appendix

JX PRESS Corporation has created and use the training template code in order to enhance team development capability and development speed.

We have created this repository by transferring only the code for training with Vertex AI from JX's training template code to [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).

For more information on JX's training template code, see [How we at JX PRESS Corporation devise for team development of R&D that tends to become a genus](https://tech.jxpress.net/entry/2021/10/27/160154) and [PyTorch Lightning explained by a heavy user](https://tech.jxpress.net/entry/2021/11/17/112214). (Now these blogs are written in Japanese. If you want to see, please translate it into your language. We would like to translate it in English and publish it someday)
<br>

# 😍 Main contributors

The transfer to this repository was done by [Yongtae](https://github.com/Yongtae723), but the development was conceived and proposed by [Yongtae](https://github.com/Yongtae723) and [near129](https://github.com/near129) led the code development.

<br>

### 🔍  What we want to improve

- Many parameters are obtained from config file in shell script, since `gcloud` command does not work as expected. But I think it is not beautiful.

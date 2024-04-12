# Miner

Miners train locally and periodically publish their best model to hugging face and commit the metadata for that model to the Bittensor chain.

Miners can only have one model associated with them on the chain for evaluation by validators at a time. The list of allowed model types by block can be found in [constants/__init__.py](https://github.com/RaoFoundation/pretraining/blob/main/constants/__init__.py#L57). Other relevant constraints are also listed in that file.

The communication between a miner and a validator happens asynchronously chain and therefore Miners do not need to be running continuously. Validators will use whichever metadata was most recently published by the miner to know which model to download from hugging face.

# System Requirements

Miners will need enough disk space to store their model as they work on. Max size of model is defined in [constants/__init__.py](https://github.com/RaoFoundation/pretraining/blob/main/constants/__init__.py#L57). It is recommended to have at least 50 GB of disk space.

Miners will need enough processing power to train their model. The device the model is trained on is recommended to be a large GPU with atleast 20 GB of VRAM.

# Getting started

## Prerequisites

1. Get a Hugging Face Account: 

Miner and validators use hugging face in order to share model state information. Miners will be uploading to hugging face and therefore must attain a account from [hugging face](https://huggingface.co/) along with a user access token which can be found by following the instructions [here](https://huggingface.co/docs/hub/security-tokens).

Make sure that any repo you create for uploading is public so that the validators can download from it for evaluation.

2. Clone the repo

```shell
git clone https://github.com/RaoFoundation/pretraining.git
```

3. Setup your python [virtual environment](https://docs.python.org/3/library/venv.html) or [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

4. Install the requirements. From your virtual environment, run
```shell
cd pretraining
python -m pip install -e .
```

Note: flash-attn may not have their dependencies set up correctly. If you run into issues try installing those requirements separately first:
```shell
pip install packaging
pip install wheel
pip install torch
```

5. Make sure you've [created a Wallet](https://docs.bittensor.com/getting-started/wallets) and [registered a hotkey](https://docs.bittensor.com/subnets/register-and-participate).

6. (Optional) Run a Subtensor instance:

Your node will run better if you are connecting to a local Bittensor chain entrypoint node rather than using Opentensor's. 
We recommend running a local node as follows and passing the ```--subtensor.network local``` flag to your running miners/validators. 
To install and run a local subtensor node follow the commands below with Docker and Docker-Compose previously installed.
```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker compose up --detach
```
---

# Running the Miner

The mining script uploads a model to hugging face which will be evaluated by validators.

See [Validator Psuedocode](docs/validator.md#validator) for more information on how they the evaluation occurs.

## Env File

The Miner requires a .env file with your hugging face access token in order to upload models.

Create a `.env` file in the `pretraining` directory and add the following to it:
```shell
HF_ACCESS_TOKEN="YOUR_HF_ACCESS_TOKEN"
```

## Starting the Miner

To start your miner the most basic command is

```shell
python neurons/miner.py --wallet.name coldkey --wallet.hotkey hotkey --hf_repo_id my-username/my-project --avg_loss_upload_threshold YOUR_THRESHOLD
```

- `--wallet.name`: should be the name of the coldkey that contains the hotkey your miner is registered with.

- `--wallet.hotkey`: should be the name of the hotkey that your miner is registered with.

- `--hf_repo_id`: should be the namespace/model_name that matches the hugging face repo you want to upload to. Must be public so that the validators can download from it.

- `--avg_loss_upload_threshold`: should be the minimum average loss before you want your miner to upload the model.


### Flags

The Miner offers some flags to customize properties, such as how to train the model and which hugging face repo to upload to.

You can view the full set of flags by running
```shell
python ./neurons/miner.py -h
```

Some flags you may find useful:

- `--offline`: when set you can run the miner without being registered and it will not attempt to upload the model.

- `--wandb_entity` + `--wandb_project`: when both flags are set the miner will log its training to the provided wandb project.

- `--device`: by default the miner will use your gpu but you can specify with this flag if you have multiple.

#### Training from pre-existing models

- `--load_best`: when set you will download and train the model from the current best miner on the network.
- `--load_uid`: when passing a uid you will download and train the model from the matching miner on the network.
- `--load_model_dir`: the path to a local model directory [saved via Hugging Face API].
- `--load_model`: the path to a safetensors file [not necessarily saved from Hugging Face API].

---

## Manually uploading a model

In some cases you may have failed to upload a model or wish to upload a model without further training.

Due to rate limiting by the Bittensor chain you may only upload a model every 20 minutes.

You can manually upload with the following command:
```shell
python scripts/upload_model.py --load_model_dir <path to model> --hf_repo_id my-username/my-project --wallet.name coldkey --wallet.hotkey hotkey
```

## Running a custom Miner

The list of allowed model types by block can be found in [constants/__init__.py](https://github.com/RaoFoundation/pretraining/blob/main/constants/__init__.py#L57)

In that file are also the constraints per block for
1. Total number of parameters.
2. Total size of the repo.
3. sequence_length parameter requierments.
4. Support for flash attention and bfloat16 requirements.

The `pretain/mining.py` file has several methods that you may find useful. Example below.

```python
import pretrain as pt
import bittensor as bt
from transformers import PreTrainedModel

# Load a model from another miner.
model: PreTrainedModel = await pt.mining.load_remote_model(uid=123, download_dir="mydir", use_bf16_and_flash=True)

# Save the model to local file.
pt.mining.save(model, "model-foo/")

# Load the model from disk.
pt.mining.load_local_model("model-foo/", use_bf16_and_flash=True)

# Publish the model for validator evaluation.
wallet = bt.wallet()
await pt.mining.push(model, repo="jdoe/my-repo", wallet=wallet)

# Get the URL to the best model
best_uid = pt.graph.best_uid()
print(await pt.mining.get_repo(best_uid))
```
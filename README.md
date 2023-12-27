<div align="center">

# **Bittensor Pretrain Subnet** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---

# Introduction

Bittensor subnet 9 rewards miners (engineers etc) for producing pretrained Foundation-Models on the Falcon Refined Web dataset. It acts like a continuous benchmark whereby miners are rewarded for attaining the best losses on randomly sampled pages of Falcon given a consistent model architecture. The reward mechanism works as follows:

    1. Miners train and periodically host trained model weights linked to their miner key as exampled by the code in neurons/miner.py.
    2. Validators run a continuous eval on the hosted models, performing the validation system outlined in neurons/validator.py and setting weights to the chain based on the performance of each miner on the Falcon dataset.
    3. The chain aggregates weights from all active validators and runs Yuma Consensus to determine the proportion of TAO emission rewarded to miners and validators. 

---
## Pretraining

Bittensor hosts multiple incentive mechanism through which miners are evaluated by validators for performing actions well. Validators perform the process of evaluation and 'set weights', which are transactions into Bittensor's [blockchain](https://polkadot.js.org/apps/?rpc=wss%3A%2F%2Fentrypoint-finney.opentensor.ai#/explorer). Each incentive mechanism in Bittensor is called a 'subnet' and has an identifier (This particular mechanism has subnet uid 9). Weights and the amount of TAO held by the validators become inputs to Bittensor's consensus mechanism called [Yuma Consensus](https://github.com/opentensor/subtensor/blob/feature/consensus-readme/docs/consensus.md). YC drives validators towards a consensus, agreement about the value of the work done by miners. The miners with the highest agreed upon scores are minted TAO, the network digital currency. You can view this information [here](https://taostats.io/subnets/netuid-9/).

Miners within this subnet are evaluated based on the number of times the model they have hosted has a lower loss than another model on the network when randomly sampling from the near infinite Falcon Refined Web pretraining dataset. To perform well, miners must attain the lowest loss on the largest number of random batches. All models are open and accessible via a [wandb project](https://wandb.ai/opentensor-dev/pretraining-subnet) and this repo contains tools for downloading them, serving them to your miner, as well as getting data from validators about which models perform best on which pages of the Falcon Dataset. Finding the best model and delta at the earliest timestamp 
ensures the most incentive.

You can view the entire validation system by reading the code in `neurons/validator.py`. Pseudocode for the validation system is as follows:
```python
    weights = zeros(256)
    while True:
        # Fetch random sample of batches to evaluate models on
        batches = get_random_sample_of_batches_from_falcon()
        
        # Fetch and or update models.
        models = get_and_update_models_from_miners()

        # Compute losses for each batch and each model
        model_losses = {}
        for model in models:
            for batch in batches:
                loss = get_loss_for_model_on_batch( model, batch )
                model_losses[ model ].append( loss )

        # Compute wins for models.
        model_wins = {}
        for model_a in models:
            for model_b in models:
                for i in len( batches )
                    # Determine if better model loss with timestamp boosting.
                    if iswin( model_losses[ model_a ][ i ], model_losses[ model_b ][ i ], timestamp_a, timestamp_b ):
                        model_wins[ model_a ] += 1
                            
        # End epoch.
        # Weights are computed based on the ratio of wins a model attains during the epoch.
        for model_i in models:
            weights[ model_i ] += model_wins[ model_i ] / sum( model_wins.values() )
        weights = softmax( weights / temperature, dim=0 )

        # Set weights on the chain.
        set_weights( weight )
```

The behaviour of `iswin( loss_a, loss_b, timestamp_a, timestamp_b)` function intentionally skews the win function to reward models which have been hosted earlier such that newer models are only better than others iff their loss is `epsilon` percent lower accoring to the following function. Currently `epsilon` is set to 1% and is a hyper parameter of the mechanism
```python

def iswin( loss_a, loss_b, timestamp_a, timestamp_b, epsilon ):
    loss_a = (1 - epsilon) * loss_a if timestamp_a < timestamp_b else loss_a
    loss_b = (1 - epsilon) * loss_b if timestamp_b < timestamp_a else loss_b
    if loss_a < loss_b: return True
    else: return False
```

It is important to note that this affects the game theoretics of the incentive landscape since miners should only update their model (thus updating their timestamp to a newer date) if they have achieved an `epsilon` better loss on average on the Falcon Refined Web dataset than their previous model. This undermines the obvious optimal strategy for miners to copy the publicly available models from other miners. They **can** and should copy other miners, but they will always obtain fewer wins compared to them until they also decrease their loss by `epsilon`.

---

## Getting Started

TL;DR:
1. [Chat](https://discord.gg/bittensor)
2. [Leaderboard](https://taostats.io/subnets/netuid-9/)
3. [Wandb Runs](https://wandb.ai/opentensor-dev/pretraining-subnet)

This repo's main conversation is carried out in the Bittensor [Discord](https://discord.gg/bittensor). Visit the 'pretraining' channel to ask questions and get real time feedback. You can view the ongoing running of the incentive mechanism, the best miners (see 'incentive'), the most inconsensus validators (see 'vtrust') using this [taostats link](https://taostats.io/subnets/netuid-9/). The table shows all 256 participant UIDs with corresponding YC stats and earnings. You can also view all current wandb runs for miners and validators [here](https://wandb.ai/opentensor-dev/pretraining-subnet).

---

## Installing

Before beginning you will need to install `pretrain` (this repo) and `bittensor`, both require atleast python3.8. We recommend using a package manager like anaconda.
```bash
git clone https://github.com/unconst/pretrain-subnet.git
cd pretrain-subnet
python -m pip install -e . 
```

Once installed correctly you can import these packages in python.
```python
import bittensor as bt
import pretrain as pt
```

And use the Bittensor CLI (for things Bittensor related, liking seeing TAO balance, transfering funds, making wallet, and viewing the network.)
```bash
btcli --help
```

--- 
## Mining Steps

1. **Get a Wandb Account**: 

Miner and validators make heavy use of weights and biases in order to share model state and validation information. Both miners and validators must attain a wandb account from [wandb](https://wandb.ai/home) along with their wandb api key which can be found by following the instructions [here](https://docs.wandb.ai/quickstart).

2. **(Optional) Run a Subtensor instance**:

Your node will run better if you are connecting to a local Bittensor chain entrypoint node rather than using Opentensor's. 
We recommend running a local node as follows and passing the ```--subtensor.network local``` flag to your running miners/validators. 
To install and run a local subtensor node follow the commands below with Docker and Docker-Compose previously installed.
```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker compose up --detach
```

3. **Create your Bittensor wallet**.

Each miners and validator requires a Bittensor coldkey and hotkey pair. To create a wallet for either your validator or miner run the following command in your terminal. Make sure to save the mnemonic for both keys and store them in a safe place.
```bash
# to create your miner/validator cold + hotkey keys.
btcli w create --wallet.name ... --wallet.hotkey ... 
btcli w list # to view your created keys.
```
Or in python
```python
import bittensor as bt
wallet = bt.wallet().create_if_non_existent()
```

4. **Register your wallet to Subnet 9**

Miner and validator wallets must be registered to the subnet 9 mechanism before they are considered active in the network and be considered avaialble for mining TAO.
There are two options. #1 Registering your walelt by `recycling` TAO to pay for entrance. To register your key run the 
following command. If you don't have any TAO message const [t,t] on Discord for a faucet to try things out, please don't scam me.
```bash
# register your cold and associated hotkey to netuid 9 using recycle.
btcli s register --wallet.name ... --wallet.hotkey ... --netuid 9
```
The second option is to run a prolonged `proof of work` or `POW` to pay for entrance into the network. You can registed your wallet using a POW using the following command:
```bash
# register your cold and associated hotkey to netuid 9 using POW
btcli s pow_register --wallet.name ... --wallet.hotkey ... --netuid 9 
```

---

## Mining

The mining script uploads a model to wandb which will be evaluated by validators and which can be viewed by visiting this [wandb project](https://wandb.ai/opentensor-dev/pretraining-subnet).

Testing the training script. Does not require registration or a wandb account:
```bash
python neurons/miner.py --wallet.name ... --wallet.hotkey ... --offline
```

Training your model from scratch and posting the model periodically as its loss decreases on Falcon:
```bash
python neurons/miner.py --wallet.name ... --wallet.hotkey ... --num_epochs 10 --pages_per_epoch 5
```

Scraping a model from an already running miner on the network by passing its uid before training:
```bash
python neurons/miner.py --wallet.name ... --wallet.hotkey ... --num_epochs 10 --pages_per_epoch 5 --load_uid ...
```

Loading the best model on the network based on its incentive.
```bash
python neurons/miner.py --wallet.name ... --wallet.hotkey ... --num_epochs 10 --pages_per_epoch 5 --load_best
```

Pass the `--device` option to select which GPU to run on.

---

## Validating

The validation script pulls runs from the wandb project and evaluates them continuously on Falcon.
Note: validation requires you have a working GPU which you pass via `--device`. In this version release/2.0.1 you need a GPU with atleast 20GB of RAM.

Test running validation:
```bash
python neurons/validator.py
    --wallet.name YOUR_WALLET_NAME
    --wallet.hotkey YOUR_WALLET_HOTKEY
    --device YOUR_CUDA DEVICE
    --wandb.off
    --offline
```

Running your validator:
```bash
python neurons/validator.py
    --wallet.name YOUR_WALLET_NAME
    --wallet.hotkey YOUR_WALLET_HOTKEY
    --device YOUR_CUDA DEVICE
```

### Auto-updates

It is essential to keep your validator up to date. There is a helper script which will automatically update your validator to the latest version. In order to use it, the repo
should be initialized and have proper reference to `origin` remote - i.e. if you cloned the repo, you're all set.

To run the auto-updater script, use the following command:
```bash
scripts/start_validator.py
    --wallet.name YOUR_WALLET_NAME
    --wallet.hotkey YOUR_WALLET_HOTKEY
    --device YOUR_CUDA_DEVICE
```

For more information about the script, refer to [script docstrings](scripts/start_validator.py).

---

## Bittensor API

The Bittensor repository comes with tooling for interfacing with the Bittensor ecosystem, creating wallets, loading state from the chain, and registering miners and validators into the mechanism. Before continuing make sure you are familiar with the main concepts of Bittensor.

```python
import bittensor as bt

# Accesses the incentive mechanism state of the pretraining 'subnet'. A subnet is a self contained consensus engine through which miners and validators agree on value creation and through which 
# TAO (bittensor's value holding token) is distributed. The metagraph is a syncable torh object which contains the state of this consensus engine at a particular block. Here we are pulling the state
# for subnet 9, which is the subnet network id, associated with this pretraining mechanism.
metagraph = bt.metagraph( 9 )
print( metagraph )

# Participants in a Bittensor consensus mechanism are defined by their wallet which contains two cryptographic keypairs, a coldkey and a hotkey. The hotkey has low security and is used to sign messages from a running miner
# while the coldkey is encrypted at all times and used to store and move TAO. The following code snippet creates a wallet on your machine with the specified coldkey name and hotkey name.
wallet = bt.wallet( name = 'cold', hotkey = 'hot' ).create_if_non_existent()
print( wallet )
```

## Pretrain API

The pretraining repo is for the easily constructing participants (miners / validators) within subnet 9 and loading and evaluating the state of the network. For instance, as a means of pushing models you have trained, or for attaining other participants' models as checkpoints for you.

Creating miners.
```python
import bittensor as bt
import pretrain as pt

# Create a mining wallet.
wallet = bt.wallet().create_if_non_existent()

# Output your mining directory.
print (f'''Wallet: {wallet}
    path: {pt.mining.path( wallet )}
    model_path: {pt.mining.model_path( wallet )}
    runidpath: {pt.mining.runidpath( wallet )}
''')

# Init or reinit the wandb run associtated with this wallet.
wandb_run = pt.mining.init( wallet )

# Load a specific model based on uid.
uid = 200
pt.graph.sync( uid, metagraph )
model = pt.graph.model( uid, device = config.device )

# Load the best model on the network based on incentive.
best_uid = pt.graph.best_uid( metagraph )
pt.graph.sync( best_uid, metagraph )
model = pt.graph.model( best_uid, device = config.device )

# Create a from scratch model.
model = pt.model.get_model()

# Save your model
pt.mining.save( wallet, model )

# Push your saved model to your wandb_run.
pt.mining.push( wallet, wandb_run )
```

The pretrain package also contains the following commands for pulling state from the network and performing validation. 

```python
import pretrain as pt
device = 'cuda'

# Pulls/Downloads model information and stores it onto your harddrive under `~/.bittensor/miners/netuid9/models/231/*`
pt.graph.sync( 231 )
pt.graph.sync( 200 )

# Print information about the recently synced uid.
print (f'''UID 231:
    timestamp: {pt.graph.timestamp( 231 )}
    run: {pt.graph.run( 231 )} 
    runid: {pt.graph.runid( 231 )}
    version: {pt.graph.version( 231 )}
    model_path: {pt.graph.model_path( 231 )}
    hotkey: {pt.graph.hotkey( 231 )}
    last_update: {pt.graph.last_update( 231 )}
''')

# Load downloaded model from harddrive to device.
model_231 = pt.graph.model( 231, device = device )
model_200 = pt.graph.model( 200, device = device )

# Attains batches from the Falcon Dataset based on pages 101
batches = list(pretrain.dataset.SubsetFalconLoader( batch_size = 2, sequence_length = 1024, pages = [ 101 ] ) )

# Evaluate the models on these batches.
losses_231 = pretrain.validation.compute_losses( model_231, batches, device = device )
losses_200 = pretrain.validation.compute_losses( model_200, batches, device = device )

# Compute wins from losses and batches.
timestamp_231 = pretrain.utils.get_timestamp_for_uid( 231 )
timestamp_200 = pretrain.utils.get_timestamp_for_uid( 200 )
for loss_231, loss_200 in list(zip( losses_231, losses_200 )):
    pretrain.validation.iswin( loss_231, loss_200, timestamp_231, timestamp_200 )
```


---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```

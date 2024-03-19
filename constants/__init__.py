import datetime as dt
from pathlib import Path
from transformers import (
    GPT2LMHeadModel,
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    GPTJForCausalLM,
)

# ---------------------------------
# Project Constants.
# ---------------------------------

__version__ = "2.2.1"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# The validator WANDB project.
WANDB_PROJECT = "pretraining-subnet"
# The uid for this subnet.
SUBNET_UID = 9
# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent
# The maximum bytes for the hugging face repo (5 Gigabyte).
MAX_HUGGING_FACE_BYTES = 5 * 1024 * 1024 * 1024
# A mapping of block numbers to the max model size as of that block.
# This dictionary must remain ordered by key.
MAX_MODEL_PARAMETER_SIZES = [
    (0, 186_000_000),
    (2_405_920, 772_000_000),
]
# The number of run steps to log to single wandb run.
MAX_RUN_STEPS_PER_WANDB_RUN = 100

# ---------------------------------
# Miner/Validator Model parameters.
# ---------------------------------

weights_version_key = __spec_version__

# validator weight moving average term
alpha = 0.5
# validator scoring exponential temperature
# 0.01 gives ~96% to best model with only ~3 receiving any weights.
temperature = 0.01
# validator score boosting for earlier models.
timestamp_epsilon = 0.005
# validators number of pages to eval over miners on each step.
n_eval_pages = 3
# validator eval batch size.
batch_size = 1
# validator eval batch min to keep for next loop.
sample_min = 10
# validator eval batch max. Difference from min is room to eval newly uploaded models.
sample_max = 25
# validator incentive threshold to prioritize updates. All incentives add up to 1.
update_priority_incentive_threshold = 0.01
# validator eval sequence length.
sequence_length = 1024
# time required between updates to the chain
chain_update_cadence = dt.timedelta(minutes=20)
# List of allowed model types.
allowed_model_types = {
    GPT2LMHeadModel,
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    GPTJForCausalLM,
}

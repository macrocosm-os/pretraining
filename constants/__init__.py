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
    PhiForCausalLM,
    GemmaForCausalLM,
)
from model.data import ModelCriteria, TokenizerIdentifier

# ---------------------------------
# Project Constants.
# ---------------------------------

# Release
__version__ = "3.2.2"

# Validator schema version
__validator_version__ = "2.2.2"
version_split = __validator_version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)

# The validator WANDB project.
WANDB_PROJECT = "pretraining-validators"

# The uid for this subnet.
SUBNET_UID = 9

# The root directory of this project.
ROOT_DIR = Path(__file__).parent.parent

# COMPETITION CHANGES
# Block at which 7b models, 4096 sequence lengths, new tokenizer, bfloat16, and flash attention are used.
BLOCK_7B = 2_786_061

# Block at which FineWeb edu score 2 dataset is used for evaluation
BLOCK_FW_EDU_SCORE_2 = 3_307_004

# FIXING MODEL CRITERIA

# Fixing sequence length
SEQUENCE_LENGTH_1 = 1024
SEQUENCE_LENGTH_2 = 4096

# Fixing evaluation dataset
DATASET_1 = "Falcon/RefinedWeb"
DATASET_2 = "HF/FineWebEdu2"


# A mapping of block numbers to the supported model types as of that block.
ALLOWED_MODEL_TYPES_1 = {
    GPT2LMHeadModel,
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    GPTJForCausalLM,
}
ALLOWED_MODEL_TYPES_2 = {
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    PhiForCausalLM,
    GemmaForCausalLM,
}


# A mapping of block numbers to ModelCriteria. Must be ordered by block.
MODEL_CRITERIA_BY_BLOCK = [
    (
        0,
        ModelCriteria(
            sequence_length=SEQUENCE_LENGTH_1,
            optimized=False,
            max_model_bytes=5 * 1024 * 1024 * 1024,
            max_model_parameters=186_000_000,
            allowed_model_types=ALLOWED_MODEL_TYPES_1,
            tokenizer_identifier=TokenizerIdentifier.DISTILGPT_2,
        ),
    ),
    (
        2_405_920,
        ModelCriteria(
            sequence_length=SEQUENCE_LENGTH_1,
            optimized=False,
            max_model_bytes=5 * 1024 * 1024 * 1024,
            max_model_parameters=772_000_000,
            allowed_model_types=ALLOWED_MODEL_TYPES_1,
            tokenizer_identifier=TokenizerIdentifier.DISTILGPT_2,
        ),
    ),
    (
        BLOCK_7B,
        ModelCriteria(
            sequence_length=SEQUENCE_LENGTH_2,
            optimized=True,
            max_model_bytes=15 * 1024 * 1024 * 1024,
            max_model_parameters=6_900_000_000,
            allowed_model_types=ALLOWED_MODEL_TYPES_2,
            tokenizer_identifier=TokenizerIdentifier.GPT_4_TIKTOKEN,
        ),
    ),
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
n_eval_pages = 18
# validator eval batch size.
batch_size = 1
# validator eval batch min to keep for next loop.
sample_min = 6
# validator eval batch max. Difference from min is room to eval newly uploaded models.
sample_max = 14
# validator incentive threshold to prioritize updates. All incentives add up to 1.
update_priority_incentive_threshold = 0.01
# time required between updates to the chain.
chain_update_cadence = dt.timedelta(minutes=20)
# time required between retrying evaluation of a stale model. (First retry will be immediate).
model_retry_cadence = dt.timedelta(hours=4)

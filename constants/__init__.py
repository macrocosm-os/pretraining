import math
import datetime as dt
from pathlib import Path

import torch
import pretrain as pt

from transformers import (
    GPT2LMHeadModel,
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    GPTJForCausalLM,
    PhiForCausalLM,
    Phi3ForCausalLM,
    GemmaForCausalLM,
    Gemma2ForCausalLM,
    Qwen2ForCausalLM,
    StableLmForCausalLM,
)

from taoverse.model.competition.data import (
    Competition,
    ModelConstraints,
    NormValidationConstraints,
)
from competitions.data import CompetitionId

from typing import Dict, List, Tuple

# ---------------------------------
# Project Constants.
# ---------------------------------

# Release
__version__ = "4.0.1"

# Validator schema version
__validator_version__ = "3.0.0"
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

# Minimum stake to consider a validator when checking for miners with weights.
# This corresponded to top-10 validator on july 31st, 2024
WEIGHT_SYNC_VALI_MIN_STAKE = 200_000

# Minimum percent of weight on a vali for a miner to be considered a top miner.
# Since there can be multiple competitions at different reward percentages we can't just check biggest.
WEIGHT_SYNC_MINER_MIN_PERCENT = 0.10

# A mapping of block numbers to the supported model types as of that block.
ALLOWED_MODEL_TYPES_1 = {
    GPT2LMHeadModel,
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    GPTJForCausalLM,
    StableLmForCausalLM,
    Phi3ForCausalLM,
    Qwen2ForCausalLM,
}
ALLOWED_MODEL_TYPES_2 = {
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    PhiForCausalLM,
    GemmaForCausalLM,
    Gemma2ForCausalLM,
    StableLmForCausalLM,
    Phi3ForCausalLM,
    Qwen2ForCausalLM,
}

# Defined dataset by competition id
DATASET_BY_COMPETITION_ID: Dict[CompetitionId, str] = {
    CompetitionId.M772_MODEL : pt.dataset.SubsetFalconLoader,
    CompetitionId.B3_MODEL : pt.dataset.SubsetFalconLoader,
    CompetitionId.B7_MODEL : pt.dataset.SubsetFineWebEdu2Loader,
}

# Defined model constraints by competition id to ensure they are constant across blocks.
MODEL_CONSTRAINTS_BY_COMPETITION_ID: Dict[CompetitionId, ModelConstraints] = {
    CompetitionId.M772_MODEL: ModelConstraints(
        max_model_parameter_size=772_000_000,
        min_model_parameter_size=572_000_000,
        sequence_length=1024,
        allowed_architectures=ALLOWED_MODEL_TYPES_1,
        tokenizer="distilgpt2",
        eval_block_delay=0,
    ),
    CompetitionId.B7_MODEL: ModelConstraints(
        max_model_parameter_size=6_900_000_000,
        min_model_parameter_size=6_700_000_000,
        sequence_length=4096,
        allowed_architectures=ALLOWED_MODEL_TYPES_2,
        tokenizer="Xenova/gpt-4",
        kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        eval_block_delay=0,
    ),
    CompetitionId.B3_MODEL: ModelConstraints(
        max_model_parameter_size=3_400_000_000,
        min_model_parameter_size=3_200_000_000,
        sequence_length=4096,
        allowed_architectures=ALLOWED_MODEL_TYPES_2,
        tokenizer="Xenova/gpt-4",
        kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        eval_block_delay=0,
    )
}


# Schedule of competitions by block.
COMPETITION_SCHEDULE_BY_BLOCK: List[Tuple[int, List[Competition]]] = [
    (
        0,
        [
            Competition(
                CompetitionId.B7_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B7_MODEL],
                1.0,
            )

        ],
    ),
    (
        3_565_190,
        [
            Competition(
                CompetitionId.M772_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.M772_MODEL],
                0.35,
            ),
            Competition(
                CompetitionId.B7_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B7_MODEL],
                0.65,
            )

        ],
    ),
]

for block_and_competitions in COMPETITION_SCHEDULE_BY_BLOCK:
    assert math.isclose(
        sum(competition.reward_percentage for competition in block_and_competitions[1]),
        1.0,
    )


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
timestamp_epsilon_experiment_start_block = 3_593_150
timestamp_epsilon_experiment = 0.001
timestamp_epsilon_experiment_weight_percent = 0.05
# validators number of pages to eval over miners on each step.
n_eval_pages = 18
# validator eval batch size.
batch_size = 1
# validator eval batch min to keep for next loop.
sample_min = 5
# Max number of uids that can be either pending eval or currently being evaluated.
updated_models_limit = 15
# validator incentive threshold to prioritize updates. All incentives add up to 1.
update_priority_incentive_threshold = 0.01
# time required between updates to the chain.
chain_update_cadence = dt.timedelta(minutes=20)
# time required between retrying evaluation of a stale model. (First retry will be immediate).
model_retry_cadence = dt.timedelta(hours=4)

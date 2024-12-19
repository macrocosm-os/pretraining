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
    GemmaForCausalLM,
    Gemma2ForCausalLM,
    Qwen2ForCausalLM,
)

from taoverse.model.competition.data import (
    Competition,
    ModelConstraints,
    NormValidationConstraints,
)
from taoverse.model.competition.epsilon import FixedEpsilon, LinearDecay
from competitions.data import CompetitionId

from typing import Dict, List, Tuple

# ---------------------------------
# Project Constants.
# ---------------------------------

# Release
__version__ = "4.6.4"

# Validator schema version
__validator_version__ = "4.6.0"
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


# Activation block for using The Stack V2 Dedup
BLOCK_STACK_V2_DEDUP = 4_453_709

# Minimum percent of weight on a vali for a miner to be considered a top miner.
# Since there can be multiple competitions at different reward percentages we can't just check biggest.
WEIGHT_SYNC_MINER_MIN_PERCENT = 0.05

# A mapping of block numbers to the supported model types as of that block.
ALLOWED_MODEL_TYPES_1 = {
    GPT2LMHeadModel,
    MistralForCausalLM,
    LlamaForCausalLM,
    BartForCausalLM,
    FalconForCausalLM,
    GPTNeoXForCausalLM,
    GPTJForCausalLM,
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
    Qwen2ForCausalLM,
}

# Defined dataset by competition id
DATASET_BY_COMPETITION_ID: Dict[CompetitionId, str] = {
    CompetitionId.B3_MODEL: pt.dataset.SubsetFalconLoader,
    CompetitionId.B7_MODEL: pt.dataset.SubsetFineWebEdu2Loader,
    CompetitionId.B14_MODEL: pt.dataset.SubsetFineWebEdu2Loader,
    # B14 model multi dataset adds the following dataset to the baseline b14 competition.
    CompetitionId.B14_MODEL_MULTI_DATASET: pt.dataset.SubsetStackV1DedupLoader,
}

# This is to be used at block for the Stack V2
DATASET_BY_COMPETITION_ID_2: Dict[CompetitionId, str] = {
    CompetitionId.B3_MODEL: pt.dataset.SubsetFalconLoader,
    CompetitionId.B7_MODEL: pt.dataset.SubsetFineWebEdu2Loader,
    CompetitionId.B14_MODEL: pt.dataset.SubsetFineWebEdu2Loader,
    # B14 model multi dataset adds the following dataset to the baseline b14 competition.
    CompetitionId.B14_MODEL_MULTI_DATASET: pt.dataset.SubsetStackV2DedupLoader,
}

# Synchronize on blocks roughly every 30 minutes.
SYNC_BLOCK_CADENCE = 150
# Delay at least as long as the sync block cadence with an additional buffer.
EVAL_BLOCK_DELAY = SYNC_BLOCK_CADENCE + 100

# Defined model constraints by competition id with decaying epsilon
MODEL_CONSTRAINTS_BY_COMPETITION_ID: Dict[CompetitionId, ModelConstraints] = {
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
        eval_block_delay=EVAL_BLOCK_DELAY,
        epsilon_func=LinearDecay(0.005, 0.0005, 50400),
        max_bytes=15 * 1024 * 1024 * 1024,
    ),
    CompetitionId.B14_MODEL: ModelConstraints(
        max_model_parameter_size=13_900_000_000,
        min_model_parameter_size=13_700_000_000,
        sequence_length=4096,
        allowed_architectures=ALLOWED_MODEL_TYPES_2,
        tokenizer="Xenova/gpt-4",
        kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        eval_block_delay=EVAL_BLOCK_DELAY,
        epsilon_func=LinearDecay(0.005, 0.0005, 50400),
        max_bytes=29 * 1024 * 1024 * 1024,
    ),
}

MODEL_CONSTRAINTS_BY_COMPETITION_ID_2: Dict[CompetitionId, ModelConstraints] = {
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
        eval_block_delay=EVAL_BLOCK_DELAY,
        epsilon_func=LinearDecay(0.005, 0.0002, 28800),
        max_bytes=15 * 1024 * 1024 * 1024,
    ),
    CompetitionId.B14_MODEL: ModelConstraints(
        max_model_parameter_size=13_900_000_000,
        min_model_parameter_size=13_700_000_000,
        sequence_length=4096,
        allowed_architectures=ALLOWED_MODEL_TYPES_2,
        tokenizer="Xenova/gpt-4",
        kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        eval_block_delay=EVAL_BLOCK_DELAY,
        epsilon_func=LinearDecay(0.005, 0.0002, 36000),
        max_bytes=29 * 1024 * 1024 * 1024,
    ),
    # This constraint is not actually used, it is added as a copy
    # of the 14B-model competition constraint entry.
    # This is just to keep the size of the constraint dict equal
    # to the number of competitions so `update_models_limit` is
    # set correctly below.
    # This hack will be removed once native support for multi datasets
    # is implemented in a future release.
    CompetitionId.B14_MODEL_MULTI_DATASET: ModelConstraints(
        max_model_parameter_size=13_900_000_000,
        min_model_parameter_size=13_700_000_000,
        sequence_length=4096,
        allowed_architectures=ALLOWED_MODEL_TYPES_2,
        tokenizer="Xenova/gpt-4",
        kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        eval_block_delay=EVAL_BLOCK_DELAY,
        epsilon_func=LinearDecay(0.005, 0.0002, 36000),
        max_bytes=29 * 1024 * 1024 * 1024,
    ),
}

# Schedule of competitions by block.
COMPETITION_SCHEDULE_BY_BLOCK: List[Tuple[int, List[Competition]]] = [
    (
        0,
        [
            Competition(
                CompetitionId.B3_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B3_MODEL],
                0.29,
            ),
            Competition(
                CompetitionId.B14_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B14_MODEL],
                0.57,
            ),
            Competition(
                CompetitionId.B14_MODEL_MULTI_DATASET,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B14_MODEL],
                0.14,
            ),
        ],
    ),
    (
        BLOCK_STACK_V2_DEDUP,
        [
            Competition(
                CompetitionId.B3_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID_2[CompetitionId.B3_MODEL],
                0.2,
            ),
            Competition(
                CompetitionId.B14_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID_2[CompetitionId.B14_MODEL],
                0.4,
            ),
            Competition(
                CompetitionId.B14_MODEL_MULTI_DATASET,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID_2[CompetitionId.B14_MODEL],
                0.4,
            ),
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

# validators number of pages to eval over miners on each step.
pages_per_eval_unpack = 10  # With sample unpacking
pages_per_eval_pack = 22

# In a future release we will update the loaders to be able to load a certain number of tokens rather than pages.
# Until then we need to set this manually
pages_per_eval_stack_v1_dedup = 1
pages_per_eval_stack_v2_dedup = 9

# validator eval batch size.
batch_size = 1
# validator eval batch min to keep for next loop.
sample_min = 5
# Max number of uids that can be either pending eval or currently being evaluated.
# We allow the sample_min per competition + 10 additional models to be held at any one time.
updated_models_limit = sample_min * len(MODEL_CONSTRAINTS_BY_COMPETITION_ID_2) + 10
# time required between updates to the chain.
chain_update_cadence = dt.timedelta(minutes=20)
# Number of blocks required between retrying evaluation of a model.
model_retry_cadence = 300  # Roughly 1 hour
# How frequently to check the models given weights by other large validators.
scan_top_model_cadence = dt.timedelta(minutes=30)

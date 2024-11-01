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
__version__ = "4.5.4"

# Validator schema version
__validator_version__ = "4.5.4"
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

# Starting block for 3B, 7B* (epsilon experiment) and sample unpacking
BLOCK_3B_7BSTAR_UNPACK = 3_601_190

# Starting block for activating sample unpacking
BLOCK_SAMPLE_PACK = 4_001_017

# Starting block for 14B* (multi dataset experiment).
BLOCK_14B_STAR = 4_252_646

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
    CompetitionId.M772_MODEL: pt.dataset.SubsetFalconLoader,
    CompetitionId.B3_MODEL: pt.dataset.SubsetFalconLoader,
    CompetitionId.B7_MODEL: pt.dataset.SubsetFineWebEdu2Loader,
    CompetitionId.B14_MODEL: pt.dataset.SubsetFineWebEdu2Loader,
    # B14 model multi dataset adds the following dataset to the baseline b14 competition.
    CompetitionId.B14_MODEL_MULTI_DATASET: pt.dataset.SubsetFalconLoader,
}

# In a future release we will update the loaders to be able to load a certain number of tokens rather than pages.
# Until then we can use this ratio to approximate the correct ratio for B14*. Token counts using Xenova/gpt-4.
ESTIMATED_TOKENS_PER_PAGE_FALCON = 55419
ESTIMATED_TOKENS_PER_PAGE_FINEWEB = 87490
# To make the additional data be 1/4 of the total we need a 1:3 ratio.
PAGE_RATIO_14B_STAR = (
    0.33 * ESTIMATED_TOKENS_PER_PAGE_FINEWEB / ESTIMATED_TOKENS_PER_PAGE_FALCON
)

# Synchronize on blocks roughly every 30 minutes.
SYNC_BLOCK_CADENCE = 150
# Delay at least as long as the sync block cadence with an additional buffer.
EVAL_BLOCK_DELAY = SYNC_BLOCK_CADENCE + 100

# Defined model constraints by competition id with decaying epsilon
MODEL_CONSTRAINTS_BY_COMPETITION_ID: Dict[CompetitionId, ModelConstraints] = {
    CompetitionId.M772_MODEL: ModelConstraints(
        max_model_parameter_size=772_000_000,
        min_model_parameter_size=572_000_000,
        sequence_length=1024,
        allowed_architectures=ALLOWED_MODEL_TYPES_1,
        tokenizer="distilgpt2",
        eval_block_delay=EVAL_BLOCK_DELAY,
        epsilon_func=LinearDecay(0.005, 0.0001, 50400),
        max_bytes=5 * 1024 * 1024 * 1024,
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
        eval_block_delay=EVAL_BLOCK_DELAY,
        epsilon_func=LinearDecay(0.005, 0.0001, 50400),
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
        epsilon_func=LinearDecay(0.005, 0.0001, 72000),
        max_bytes=29 * 1024 * 1024 * 1024,
    ),
}

# Schedule of competitions by block.
COMPETITION_SCHEDULE_BY_BLOCK: List[Tuple[int, List[Competition]]] = [
    (
        0,
        [
            Competition(
                CompetitionId.M772_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.M772_MODEL],
                0.14,
            ),
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
        ],
    ),
    (
        BLOCK_14B_STAR,
        [
            Competition(
                CompetitionId.B3_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B3_MODEL],
                0.29,
            ),
            Competition(
                CompetitionId.B14_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B14_MODEL],
                0.66,
            ),
            # TODO decide specific weight.
            Competition(
                CompetitionId.B14_MODEL_MULTI_DATASET,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B14_MODEL],
                0.05,
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

# block to activate sample packing
sample_pack_block = BLOCK_SAMPLE_PACK

# validators number of pages to eval over miners on each step.
pages_per_eval_unpack = 5  # With sample unpacking
pages_per_eval_pack = 11

# validator eval batch size.
batch_size = 1
# validator eval batch min to keep for next loop.
sample_min = 5
# Max number of uids that can be either pending eval or currently being evaluated.
# We allow the sample_min per competition + 10 additional models to be held at any one time.
updated_models_limit = sample_min * len(MODEL_CONSTRAINTS_BY_COMPETITION_ID) + 10
# time required between updates to the chain.
chain_update_cadence = dt.timedelta(minutes=20)
# Number of blocks required between retrying evaluation of a model.
model_retry_cadence = 300  # Roughly 1 hour
# How frequently to check the models given weights by other large validators.
scan_top_model_cadence = dt.timedelta(minutes=30)

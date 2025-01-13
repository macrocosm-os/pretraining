import datetime as dt
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from taoverse.model.competition.data import Competition, ModelConstraints
from taoverse.model.competition.epsilon import LinearDecay
from taoverse.model.eval.normalization import NormalizationId
from taoverse.model.eval.task import EvalTask
from transformers import (
    BartForCausalLM,
    FalconForCausalLM,
    Gemma2ForCausalLM,
    GemmaForCausalLM,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
    MistralForCausalLM,
    PhiForCausalLM,
    Qwen2ForCausalLM,
)

import pretrain as pt
from competitions.data import CompetitionId
from pretrain.datasets.ids import DatasetId
from pretrain.eval.method import EvalMethodId

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

# Minimum percent of weight on a vali for a miner to be considered a top miner.
# Since there can be multiple competitions at different reward percentages we can't just check biggest.
WEIGHT_SYNC_MINER_MIN_PERCENT = 0.05

# Validator eval batch size.
BATCH_SIZE = 1
# Validators number of samples to eval over miners on each step.
# TODO: Confirm 400 is good and if we should use smaller numbers for multi-dataset competitions.
SAMPLES_PER_EVAL = 400

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

DATASET_BY_COMPETITION_ID: Dict[CompetitionId, str] = {
    CompetitionId.B3_MODEL: pt.dataset.SubsetFalconLoader,
    CompetitionId.B14_MODEL: pt.dataset.SubsetFineWebEdu2Loader,
    # B14 model multi dataset adds the following dataset to the baseline b14 competition.
    CompetitionId.B14_MODEL_MULTI_DATASET: pt.dataset.SubsetStackV2DedupLoader,
}

# Synchronize on blocks roughly every 30 minutes.
SYNC_BLOCK_CADENCE = 150
# Delay at least as long as the sync block cadence with an additional buffer.
EVAL_BLOCK_DELAY = SYNC_BLOCK_CADENCE + 100

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
                0.2,
                eval_tasks=[
                    EvalTask(
                        name="FALCON",
                        method_id=EvalMethodId.TEXT_LOSS,
                        dataset_id=DatasetId.FALCON,
                        normalization_id=NormalizationId.NONE,
                        dataset_kwargs={
                            "batch_size": BATCH_SIZE,
                            "num_samples": SAMPLES_PER_EVAL,
                        },
                        weight=1,
                    ),
                ],
            ),
            Competition(
                CompetitionId.B14_MODEL,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B14_MODEL],
                0.4,
                eval_tasks=[
                    EvalTask(
                        name="FINEWEB",
                        method_id=EvalMethodId.TEXT_LOSS,
                        dataset_id=DatasetId.FINEWEB,
                        normalization_id=NormalizationId.NONE,
                        dataset_kwargs={
                            "batch_size": BATCH_SIZE,
                            "num_samples": SAMPLES_PER_EVAL,
                        },
                        weight=1,
                    ),
                ],
            ),
            # This competition is currently being run as part of B14.
            # In a future block this eval task definition will become the new B14 competition eval.
            Competition(
                CompetitionId.B14_MODEL_MULTI_DATASET,
                MODEL_CONSTRAINTS_BY_COMPETITION_ID[CompetitionId.B14_MODEL],
                0.4,
                eval_tasks=[
                    EvalTask(
                        name="FINEWEB",
                        method_id=EvalMethodId.TEXT_LOSS,
                        dataset_id=DatasetId.FINEWEB,
                        normalization_id=NormalizationId.NONE,
                        dataset_kwargs={
                            "batch_size": BATCH_SIZE,
                            "num_samples": SAMPLES_PER_EVAL,
                        },
                        weight=0.85,
                    ),
                    EvalTask(
                        name="STACKV2",
                        method_id=EvalMethodId.TEXT_LOSS,
                        dataset_id=DatasetId.STACK2,
                        normalization_id=NormalizationId.NONE,
                        dataset_kwargs={
                            "batch_size": BATCH_SIZE,
                            "num_samples": SAMPLES_PER_EVAL,
                        },
                        weight=0.15,
                    ),
                ],
            ),
        ],
    ),
]

for block_and_competitions in COMPETITION_SCHEDULE_BY_BLOCK:
    assert math.isclose(
        sum(competition.reward_percentage for competition in block_and_competitions[1]),
        1.0,
    )
    for comp in block_and_competitions[1]:
        assert math.isclose(
            sum(task.weight for task in comp.eval_tasks),
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

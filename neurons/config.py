import argparse
import os
import bittensor as bt
import torch
import constants


def validator_config():
    """Returns the config for the validator."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device name.",
    )
    parser.add_argument(
        "--wandb.off",
        dest="wandb.on",
        action="store_false",
        help="Turn off wandb logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=constants.WANDB_PROJECT,
        help="The wandb project to log to.",
    )
    parser.add_argument(
        "--blocks_per_epoch",
        type=int,
        default=50,
        help="Number of blocks to wait before setting weights.",
    )
    parser.add_argument(
        "--pages_per_eval",
        type=int,
        default=None,
        help="Number of pages used to eval each step. If not specified, it will be automatically set.",
    )
    parser.add_argument(
        "--sample_min",
        type=int,
        default=constants.sample_min,
        help="Number of uids to bring to next eval.",
    )
    parser.add_argument(
        "--updated_models_limit",
        type=int,
        default=constants.updated_models_limit,
        help="Max number of uids that can be either pending eval or currently being evaluated.",
    )
    parser.add_argument(
        "--dont_set_weights",
        action="store_true",
        help="Validator does not set weights on the chain.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Does not launch a wandb run, does not set weights, does not check that your key is registered.",
    )
    parser.add_argument(
        "--model_dir",
        default=os.path.join(constants.ROOT_DIR, "model-store/"),
        help="Where to store downloaded models",
    )
    parser.add_argument(
        "--netuid",
        type=str,
        default=constants.SUBNET_UID,
        help="The subnet UID.",
    )

    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    return config

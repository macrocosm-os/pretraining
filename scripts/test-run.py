import abc
from argparse import ArgumentParser
import asyncio
import time
from typing import Dict, List, Tuple
import requests
import wandb
import torch
import random
from tqdm import tqdm
from model.data import ModelMetadata, TokenizerIdentifier
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
import pretrain as pt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
from collections import defaultdict
import os
import wandb
import pandas as pd
from dotenv import load_dotenv
import bittensor as bt
import constants
import model.utils as model_utils

if __name__ == "__main__":
    bt.logging()

    parser = ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default=None)

    bt.subtensor.add_args(parser)

    config = bt.config(parser=parser)
    args = parser.parse_args()

    # Download model (mostly) just like validators do
    bt.logging.info("Downloading model")

    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=constants.SUBNET_UID)
    best_uid = pt.graph.best_uid(metagraph=metagraph)
    hotkey = metagraph.hotkeys[173]

    metagraph_store = ChainModelMetadataStore(subtensor)
    metadata = asyncio.run(metagraph_store.retrieve_model_metadata(hotkey))

    store = HuggingFaceModelStore()
    model = asyncio.run(store.download_model(metadata.id, args.cache_dir))

    # Retrieve local just like validators do
    model_i = self.local_store.retrieve_model(
        hotkey,
        model_i_metadata.id,
        optimized,
    )

    # Compute losses just like validators do

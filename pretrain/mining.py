# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

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

import os
import sys
import time
from typing import Optional
import constants
from model.data import Model, ModelId
from model.storage.chain.chain_model_metadata_store import ChainModelMetadataStore
from model.storage.hugging_face.hugging_face_model_store import HuggingFaceModelStore
from model.storage.model_metadata_store import ModelMetadataStore
from model.storage.remote_model_store import RemoteModelStore
import bittensor as bt
from transformers import PreTrainedModel, AutoModelForCausalLM
import pretrain as pt
from safetensors.torch import load_model

from utilities import utils
import torch


def model_path(base_dir: str, run_id: str) -> str:
    """
    Constructs a file path for storing the model relating to a training run.
    """
    return os.path.join(base_dir, "training", run_id)


async def push(
    model: PreTrainedModel,
    repo: str,
    wallet: bt.wallet,
    retry_delay_secs: int = 60,
    metadata_store: Optional[ModelMetadataStore] = None,
    remote_model_store: Optional[RemoteModelStore] = None,
    use_bf16_and_flash=True,
):
    """Pushes the model to Hugging Face and publishes it on the chain for evaluation by validators.

    Args:
        model (PreTrainedModel): The model to push.
        repo (str): The repo to push to. Must be in format "namespace/name".
        wallet (bt.wallet): The wallet of the Miner uploading the model.
        retry_delay_secs (int): The number of seconds to wait before retrying to push the model to the chain.
        metadata_store (Optional[ModelMetadataStore]): The metadata store. If None, defaults to writing to the
            chain.
        remote_model_store (Optional[RemoteModelStore]): The remote model store. If None, defaults to writing to HuggingFace
        use_bf16_and_flash (bool): If the model should be uploaded using bfloat16 and flash attention 2.
    """
    bt.logging.info("Pushing model")

    if metadata_store is None:
        metadata_store = ChainModelMetadataStore(bt.subtensor(), wallet)

    if remote_model_store is None:
        remote_model_store = HuggingFaceModelStore()

    # First upload the model to HuggingFace.
    namespace, name = utils.validate_hf_repo_id(repo)
    model_id = ModelId(namespace=namespace, name=name)
    model_id = await remote_model_store.upload_model(
        Model(id=model_id, pt_model=model), use_bf16_and_flash
    )

    bt.logging.success(
        f"Uploaded model to hugging face. Now committing to the chain with model_id: {model_id}"
    )

    # We can only commit to the chain every 20 minutes, so run this in a loop, until
    # successful.
    while True:
        try:
            await metadata_store.store_model_metadata(
                wallet.hotkey.ss58_address, model_id
            )

            bt.logging.info(
                "Wrote model metadata to the chain. Checking we can read it back..."
            )

            model_metadata = await metadata_store.retrieve_model_metadata(
                wallet.hotkey.ss58_address
            )

            if not model_metadata or model_metadata.id != model_id:
                bt.logging.error(
                    f"Failed to read back model metadata from the chain. Expected: {model_id}, got: {model_metadata}"
                )
                raise ValueError(
                    f"Failed to read back model metadata from the chain. Expected: {model_id}, got: {model_metadata}"
                )

            bt.logging.success("Committed model to the chain.")
            break
        except Exception as e:
            bt.logging.error(f"Failed to advertise model on the chain: {e}")
            bt.logging.error(f"Retrying in {retry_delay_secs} seconds...")
            time.sleep(retry_delay_secs)


def save(model: PreTrainedModel, model_dir: str):
    """Saves a model to the provided directory"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Save the model state to the specified path.
    model.save_pretrained(
        save_directory=model_dir,
        safe_serialization=True,
    )


async def get_repo(
    uid: int,
    metagraph: Optional[bt.metagraph] = None,
    metadata_store: Optional[ModelMetadataStore] = None,
) -> str:
    """Returns a URL to the HuggingFace repo of the Miner with the given UID."""
    if metadata_store is None:
        metadata_store = ChainModelMetadataStore(bt.subtensor())
    if metagraph is None:
        metagraph = bt.metagraph(netuid=constants.SUBNET_UID)
    hotkey = metagraph.hotkeys[uid]
    model_metadata = await metadata_store.retrieve_model_metadata(hotkey)

    if not model_metadata:
        raise ValueError(f"No model metadata found for miner {uid}")

    return utils.get_hf_url(model_metadata)


def load_gpt2_model(model_file: str) -> PreTrainedModel:
    """For loading GPT2 models from the previous version of this subnet."""
    model = pt.model.get_model()
    load_model(model, model_file)
    return model


def load_local_model(model_dir: str, use_bf16_and_flash: bool) -> PreTrainedModel:
    """Loads a model from a directory."""
    if use_bf16_and_flash:
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            local_files_only=True,
            use_safetensors=True,
        )


async def load_best_model(download_dir: str, use_bf16_and_flash: bool):
    """Loads the model from the best performing miner to download_dir"""
    best_uid = pt.graph.best_uid()
    return await load_remote_model(
        best_uid, download_dir, use_bf16_and_flash=use_bf16_and_flash
    )


async def load_remote_model(
    uid: int,
    download_dir: str,
    use_bf16_and_flash: bool,
    metagraph: Optional[bt.metagraph] = None,
    metadata_store: Optional[ModelMetadataStore] = None,
    remote_model_store: Optional[RemoteModelStore] = None,
) -> PreTrainedModel:
    """Loads the model currently being advertised by the Miner with the given UID.

    Args:
        uid (int): The UID of the Miner who's model should be downloaded.
        download_dir (str): The directory to download the model to.
        use_bf16_and_flash (bool): If the model should be loaded using bfloat16 and flash attention 2.
        metagraph (Optional[bt.metagraph]): The metagraph of the subnet.
        metadata_store (Optional[ModelMetadataStore]): The metadata store. If None, defaults to reading from the
        remote_model_store (Optional[RemoteModelStore]): The remote model store. If None, defaults to reading from HuggingFace
    """

    if metagraph is None:
        metagraph = bt.metagraph(netuid=constants.SUBNET_UID)

    if metadata_store is None:
        metadata_store = ChainModelMetadataStore(subtensor=bt.subtensor())

    if remote_model_store is None:
        remote_model_store = HuggingFaceModelStore()

    hotkey = metagraph.hotkeys[uid]
    model_metadata = await metadata_store.retrieve_model_metadata(hotkey)
    if not model_metadata:
        raise ValueError(f"No model metadata found for miner {uid}")

    bt.logging.success(f"Fetched model metadata: {model_metadata}")
    model: Model = await remote_model_store.download_model(
        model_metadata.id, download_dir, use_bf16_and_flash=use_bf16_and_flash
    )
    return model.pt_model

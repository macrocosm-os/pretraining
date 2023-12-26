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
import time
import json
import wandb
import torch
import typing
import pretrain
import bittensor as bt
from datetime import datetime
from safetensors import safe_open
from safetensors.torch import load_model, save_model


def best_uid(metagraph: typing.Optional[bt.metagraph] = None):
    if not metagraph:
        metagraph = bt.subtensor().metagraph(pretrain.NETUID)
    return max(range(256), key=lambda uid: metagraph.I[uid].item())


def best_model(metagraph: typing.Optional[bt.metagraph] = None, device: str = "cpu "):
    if not metagraph:
        metagraph = bt.subtensor().metagraph(pretrain.NETUID)
    _best_uid = best_uid(metagraph)
    sync(_best_uid, metagraph=metagraph)
    return model(_best_uid, device=device)


def timestamp(uid: int) -> typing.Optional[int]:
    try:
        return metadata(uid)["timestamp"]
    except Exception as e:
        bt.logging.debug(
            "Failed to get timestamp for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} )".format(
                uid, uid
            )
        )
        return None


def run(uid: int) -> typing.Optional["wandb.run"]:
    try:
        api = wandb.Api(timeout=100)
        run = api.run(
            f"opentensor-dev/{pretrain.WANDB_PROJECT}/{metadata(uid)['runid']}"
        )
        return run
    except Exception as e:
        bt.logging.debug(
            "Failed to get run for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} )".format(
                uid, uid
            )
        )
        return None


def runid(uid: int) -> typing.Optional[str]:
    try:
        return metadata(uid)["runid"]
    except Exception as e:
        bt.logging.debug(
            "Failed to get runid for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} )".format(
                uid, uid
            )
        )
        return None


def version(uid: int) -> typing.Optional[str]:
    try:
        return metadata(uid)["version"]
    except Exception as e:
        bt.logging.debug(
            "Failed to get version for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} )".format(
                uid, uid
            )
        )
        return None


def path(uid: int) -> typing.Optional[str]:
    try:
        return os.path.dirname(metadata(uid)["model_path"])
    except Exception as e:
        bt.logging.debug(
            "Failed to get model_path for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} )".format(
                uid, uid
            )
        )
        return None


def last_download(uid: int) -> typing.Optional[float]:
    try:
        _metadata = metadata(uid)
        if "last_download" not in _metadata:
            return _metadata["last_update"]
        else:
            return _metadata["last_download"]
    except Exception as e:
        bt.logging.debug(
            "Failed to get last_download for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} ) with error: { }".format(
                uid, uid, e
            )
        )
        return None


def model_path(uid: int) -> typing.Optional[str]:
    try:
        return metadata(uid)["model_path"]
    except Exception as e:
        bt.logging.debug(
            "Failed to get model_path for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} )".format(
                uid, uid
            )
        )
        return None


def is_synced(uid: int) -> typing.Optional[str]:
    try:
        if metadata(uid) == None:
            return False
        _model_path = model_path(uid)
        if _model_path == None:
            return False
        return os.path.exists(_model_path)
    except Exception as e:
        bt.logging.debug(f"Failed to get is_synced for uid: {uid}")
        return None


def hotkey(uid: int) -> typing.Optional[str]:
    try:
        return metadata(uid)["hotkey"]
    except Exception as e:
        bt.logging.debug(
            "Failed to get hotkey for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} )".format(
                uid, uid
            )
        )
        return None


def last_update(uid: int) -> typing.Optional[float]:
    try:
        return metadata(uid)["last_update"]
    except Exception as e:
        bt.logging.debug(
            "Failed to get last_update for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} )".format(
                uid, uid
            )
        )
        return None


def has_valid_run(
    uid: int, metagraph: typing.Optional[bt.metagraph] = None
) -> typing.Optional[bool]:
    try:
        return check_run_validity(run(uid), metagraph=metagraph)[0]
    except Exception as e:
        bt.logging.debug(
            "Failed to get last_update for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} )".format(
                uid, uid
            )
        )
        return None


def metadata(uid: int):
    """
        Retrieves the file path to the model checkpoint for uid with associated timestamps and verion.
    Args:
        uid: Uid to find metadata on.
    """
    try:
        # Fill metadata from files and check if we can load weights.
        model_dir = os.path.join(pretrain.netuid_dir, "models", str(uid))
        try:
            with open(os.path.join(model_dir, "metadata.json"), "r") as f:
                meta = json.load(f)
        except Exception as e:
            # bt.logging.trace(f'Failed Metadata: uid:{uid}, no file under path:{model_dir}')
            return None
        if "version" not in meta or "timestamp" not in meta or "runid" not in meta:
            bt.logging.debug(f"Failed Metadata: uid:{uid}, metadata file corrupted.")
            return None
        else:
            # Valid metadata.
            return meta
    except Exception as e:
        # Skip this UID if any error occurs during loading of model or timestamp.
        bt.logging.debug(
            f"Failed Metadata: uid:{uid}, metadata could not be loaded with error:{e}"
        )
        return None


def model(uid: int, device: str = "cpu") -> typing.Optional[torch.nn.Module]:
    """
    Retrieves the model for a specific user ID (uid) without updating it.
    This function loads the model weights into the model.

    Parameters:
    uid (int): The user ID for which the model is to be retrieved.
    device (str): The device on which the model is to be loaded. Defaults to 'cpu'.

    Returns:
    torch.nn.Module or None: The model if successful, None otherwise.
    """
    try:
        model = pretrain.model.get_model()
        model_meta = metadata(uid)
        load_model(model, model_meta["model_path"])
        return model
    except Exception as e:
        bt.logging.debug(
            f"Error loading model for uid: {uid} with error: {e}, try pulling data for this uid with pretrain.graph.pull( {uid} )"
        )
        return None


def sync(uid: int, metagraph: typing.Optional[bt.metagraph] = None) -> bool:
    """
    Updates the model for a given user ID (uid) if there is a newer valid run.

    This function checks for the latest valid run for the specified uid and updates the local model files
    if there is a newer version available. It manages the metadata and model files based on the latest run's
    information. If no valid run is found, it cleans up any existing model and metadata files.

    Parameters:
        uid (int): The user ID for which the model update is to be checked and performed.
        metagraph (Optional[bt.metagraph]): The metagraph to use for finding the latest valid run.
                                        If not provided, it's fetched using pretrain.NETUID.

    Returns:
        bool: True if the model was updated, False otherwise.

    Note:
        - The function assumes the existence of specific directories and file paths as defined in 'pretrain'.
        - It manages model files in a directory structure based on the uid.
    """
    # Get the latest valid run for this uid
    if not metagraph:
        metagraph = bt.subtensor().metagraph(pretrain.NETUID)

    latest_valid_run = get_run_for_uid(uid, metagraph=metagraph)
    expected_hotkey = metagraph.hotkeys[uid]

    # Paths for model and metadata
    models_dir = os.path.join(pretrain.netuid_dir, "models", str(uid))
    metadata_file = os.path.join(models_dir, "metadata.json")
    model_path = os.path.join(models_dir, "model.safe")

    # If no valid runs, delete existing model and metadata
    if latest_valid_run is None:
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        if os.path.exists(model_path):
            os.remove(model_path)
            bt.logging.debug(f"Deleting {uid} model with no run.")
        return False

    # Create model directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Load model artifact and get timestamp
    model_artifact = latest_valid_run.file("model.safe")
    heartbeat = int(
        datetime.strptime(
            latest_valid_run._attrs["heartbeatAt"], "%Y-%m-%dT%H:%M:%S"
        ).timestamp()
    )
    current_meta = metadata(uid)

    # Update metadata file
    with open(metadata_file, "w") as f:
        json.dump(
            {
                "timestamp": heartbeat,
                "runid": latest_valid_run.id,
                "model_path": model_path,
                "version": latest_valid_run.config["version"],
                "blacklisted": False,
                "last_update": time.time(),
                "uid": uid,
                "hotkey": expected_hotkey,
                "last_download": current_meta["last_download"]
                if current_meta != None and "last_download" in current_meta
                else heartbeat,
            },
            f,
        )

    # Check if model needs updating
    if current_meta is not None and heartbeat <= current_meta["last_download"]:
        bt.logging.trace(
            f"Model is up to date: uid: {uid}, under path: {models_dir}, with heartbeat: {heartbeat}"
        )
        return False
    else:
        # Download the updated model artifact
        model_artifact.download(replace=True, root=models_dir)
        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "timestamp": heartbeat,
                    "runid": latest_valid_run.id,
                    "model_path": model_path,
                    "version": latest_valid_run.config["version"],
                    "blacklisted": False,
                    "last_update": time.time(),
                    "uid": uid,
                    "hotkey": expected_hotkey,
                    "last_download": heartbeat,
                },
                f,
            )
        bt.logging.debug(
            f"Updated model: uid: {uid}, under path: {models_dir}, with heartbeat: {heartbeat}"
        )
        return True


def push(
    uid, model: torch.nn.Module, path: str = os.path.expanduser("~/tmp/model.safe")
):
    """
    Saves the state of a given model and updates the corresponding Weights & Biases (wandb) run with the model file.

    This function serializes the state of the provided PyTorch model and saves it to a specified file path.
    It then uses the wandb API to log this file to the associated run, allowing for tracking and versioning of model states in wandb.

    Parameters:
        model (torch.nn.Module): The PyTorch model whose state is to be saved.
        run (wandb.run): The wandb run to which the model state will be logged.
        path (str): The file path where the model state will be saved. Defaults to '~/tmp/model.safe'.

    Note:
        - The function does not perform any checks on the validity of the provided model or wandb run.
        - The default save path is in the user's home directory under 'tmp', which should be verified or changed based on the system setup.
    """
    run = get_run_for_uid(uid)
    # Create the directory for the model path if it does not exist
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save the model state to the specified path
    save_model(model, path)
    # Log the saved model file to wandb run.
    run.save(path)


def check_run_validity(
    run: "wandb.run", metagraph: typing.Optional[bt.metagraph] = None
) -> typing.Tuple[bool, str]:
    """
        Checks the validity of a Weights & Biases (wandb) run against a metagraph.

        This function verifies the integrity and authenticity of a wandb run by checking its hotkey and signature
        against the provided metagraph. It also validates the presence of a model artifact file ('model.safe')
        and its timestamp.

    Parameters:
        run (wandb.run): The wandb run to be validated.
        metagraph: The metagraph against which the run's validity is checked.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating the validity of the run and a string message
                      explaining the validity status.

    Note:
        - The function requires the hotkey and signature to be present in the run's configuration.
        - It assumes the presence of 'model.safe' as a key artifact in the run.
        - The function catches and handles exceptions, providing an appropriate message in case of failure.
    """
    if not metagraph:
        metagraph = bt.subtensor().metagraph(pretrain.NETUID)
    try:
        # Extract hotkey and signature from the run's configuration
        hotkey = run.config["hotkey"]
        signature = run.config["signature"]

        # Check if the hotkey is registered in the metagraph
        if hotkey not in metagraph.hotkeys:
            return (
                False,
                f"Failed Signature: The hotkey: {hotkey} is not in the metagraph.",
            )

        # Verify the signature using the hotkey
        if not bt.Keypair(ss58_address=hotkey).verify(run.id, bytes.fromhex(signature)):
            return False, f"Failed Signature: The signature: {signature} is not valid"

        # Attempt to access the model artifact file
        try:
            model_artifact = run.file("model.safe")
        except:
            return False, f"Failed Signature: Does not have a model.safe file"

        # Check and convert the updated at timestamp
        try:
            int(
                datetime.strptime(
                    model_artifact.updatedAt, "%Y-%m-%dT%H:%M:%S"
                ).timestamp()
            )
        except:
            return False, f"Failed validity: Does not have a valid model.safe file"

        # The run has a valid signature
        return True, f"Passed Validity check."
    except Exception as e:
        # Handle exceptions during the validity check
        return (
            False,
            f"Failed Signature: An exception occurred while checking the signature with error: {e}",
        )


def check_run_exists(uid, metadata: dict, metagraph):
    try:
        expected_runid = metadata["runid"]
        api = wandb.Api(timeout=100)
        run = api.run(f"opentensor-dev/{pretrain.WANDB_PROJECT}/{expected_runid}")
        assert run.config["uid"] == uid
        assert run.config["hotkey"] == metagraph.hotkeys[uid]
        return True
    except Exception as e:
        bt.logging.debug(f"Check run failed with error: {e}")
        return False


def get_run_for_uid(
    uid: int, metagraph: typing.Optional[bt.metagraph] = None
) -> typing.Optional["wandb.run"]:
    """
    Retrieves a specific Weights & Biases (wandb) run for a given user ID (uid).

    This function queries the wandb API for runs associated with a specific user ID.
    It filters runs based on the version, type, run name pattern, and hotkey corresponding to the UID.
    The function returns the most recent valid run for the given UID.

    Parameters:
        uid (int): The user ID for which the wandb run is to be retrieved.

    Returns:
        Optional[wandb.run]: The most recent valid wandb run for the given UID, or None if no valid run is found.

    Note:
    - This function assumes that the metagraph and wandb API settings are correctly configured.
    - It relies on specific configuration fields (version, type, run_name, hotkey) being present in the wandb run's metadata.
    """
    if not metagraph:
        metagraph = bt.subtensor().metagraph(pretrain.NETUID)
    api = wandb.Api(timeout=100)
    expected_hotkey = metagraph.hotkeys[uid]

    # Retrieve runs from the wandb project for this uid
    runs = api.runs(
        f"opentensor-dev/{pretrain.WANDB_PROJECT}",
        filters={
            "config.version": pretrain.__version__,
            "config.type": "miner",
            "config.run_name": {"$regex": f"miner-{uid}-.*"},
            "config.hotkey": expected_hotkey,
        },
    )

    # Iterate through runs. Newer runs are processed first.
    for run in runs:
        if check_run_validity(run, metagraph=metagraph)[0]:
            return run
        else:
            continue

    return None

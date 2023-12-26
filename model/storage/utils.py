import os
from model.data import ModelId


# TODO make this configurable.
def get_local_miner_dir(hotkey: str) -> str:
    return os.path.join("local-models", hotkey)


def get_local_model_dir(hotkey: str, model_id: ModelId) -> str:
    return os.path.join(get_local_miner_dir(hotkey), model_id.path, model_id.name)

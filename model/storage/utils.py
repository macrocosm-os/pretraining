import os
from model.data import ModelId


# TODO make this configurable.
def get_local_miner_dir(uid: int) -> str:
    return os.path.join("local-models", str(uid))


def get_local_model_dir(uid: int, model_id: ModelId) -> str:
    return os.path.join(get_local_miner_dir(uid), model_id.path, model_id.name)

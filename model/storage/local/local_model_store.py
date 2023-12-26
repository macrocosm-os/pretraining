import asyncio
import shutil
from model.data import Model, ModelId
from model.storage import utils
from model.storage.model_store import ModelStore
from transformers import AutoModel, DistilBertModel, DistilBertConfig


class LocalModelStore(ModelStore):
    """Local storage based implementation for storing and retrieving a model."""

    async def clear_miner_directory(self, uid: int):
        """Clears out the directory for a given uid."""
        shutil.rmtree(path=utils.get_local_miner_dir(uid), ignore_errors=True)

    async def clear_model_directory(self, uid: int, model_id: ModelId):
        """Clears out the directory for a given model."""
        shutil.rmtree(path=utils.get_local_model_dir(uid, model_id), ignore_errors=True)

    async def store_model(self, uid: int, model: Model) -> ModelId:
        """Stores a trained model locally."""

        model.pt_model.save_pretrained(
            save_directory=utils.get_local_model_dir(uid, model.id),
            safe_serialization=True,
        )

        # Return the same model id used as we do not edit the commit information.
        return model.id

    # TODO actually make this asynchronous with threadpools etc.
    async def retrieve_model(self, uid: int, model_id: ModelId) -> Model:
        """Retrieves a trained model locally."""

        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=utils.get_local_model_dir(uid, model_id),
            revision=model_id.commit,
            local_files_only=True,
            use_safetensors=True,
        )

        return Model(id=model_id, pt_model=model)


async def test_roundtrip_model():
    """Verifies that the LocalModelStore can roundtrip a model."""
    model_id = ModelId(
        path="TestPath",
        name="TestModel",
        hash="TestHash1",
        commit="main",
    )

    pt_model = DistilBertModel(
        config=DistilBertConfig(
            vocab_size=256, n_layers=2, n_heads=4, dim=100, hidden_dim=400
        )
    )

    model = Model(id=model_id, pt_model=pt_model)
    local_model_store = LocalModelStore()

    # Clear the local storage
    await local_model_store.clear_model_directory(uid=0, model_id=model_id)

    # Store the model locally.
    await local_model_store.store_model(uid=0, model=model)

    # Retrieve the model locally.
    retrieved_model = await local_model_store.retrieve_model(uid=0, model_id=model_id)

    # Check that they match.
    # TODO create appropriate equality check.
    print(
        f"Finished the roundtrip and checking that the models match: {str(model) == str(retrieved_model)}"
    )


if __name__ == "__main__":
    asyncio.run(test_roundtrip_model())

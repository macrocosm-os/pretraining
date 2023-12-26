import asyncio
import bittensor as bt
import os
from model.data import Model, ModelId
from model.storage import utils
from model.storage.model_store import ModelStore
from transformers import AutoModel, DistilBertModel, DistilBertConfig


class HuggingFaceModelStore(ModelStore):
    """Hugging Face based implementation for storing and retrieving a model."""

    async def store_model(self, uid: int, model: Model):
        """Stores a trained model in Hugging Face."""
        # TODO error message if not found.
        token = os.getenv("HF_ACCESS_TOKEN")
        if not token:
            bt.logging.error("No Hugging Face access token found to write to the hub.")

        # PreTrainedModel.save_pretrained only saves locally
        model.pt_model.push_to_hub(
            repo_id=model.id.path + "/" + model.id.name, token=token
        )

    # TODO actually make this asynchronous with threadpools etc.
    async def retrieve_model(self, uid: int, model_id: ModelId) -> Model:
        """Retrieves a trained model from Hugging Face."""
        # Transformers library can pick up a model based on the hugging face path (username/model) + rev.
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_id.path + "/" + model_id.name,
            revision=model_id.commit,
            cache_dir=utils.get_local_model_dir(uid, model_id),
        )

        return Model(id=model_id, pt_model=model)


async def test_roundtrip_model():
    """Verifies that the HuggingFaceModelStore can roundtrip a model in hugging face."""
    hf_name = os.getenv("HF_NAME")
    model_id = ModelId(
        path=hf_name,
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
    hf_model_store = HuggingFaceModelStore()

    # Store the model in hf.
    await hf_model_store.store_model(uid=0, model=model)

    # Retrieve the model from hf.
    retrieved_model = await hf_model_store.retrieve_model(uid=0, model_id=model_id)

    # Check that they match.
    # TODO create appropriate equality check.
    print(
        f"Finished the roundtrip and checking that the models match: {str(model) == str(retrieved_model)}"
    )


async def test_retrieve_model():
    """Verifies that the HuggingFaceModelStore can retrieve a model."""
    model_id = ModelId(
        path="pszemraj",
        name="distilgpt2-HC3",
        hash="TestHash1",
        commit="6f9ad47",
    )

    hf_model_store = HuggingFaceModelStore()

    # Retrieve the model from hf (first run) or cache.
    model = await hf_model_store.retrieve_model(uid=0, model_id=model_id)

    print(f"Finished retrieving the model with id: {model.id}")


if __name__ == "__main__":
    asyncio.run(test_retrieve_model())
    # asyncio.run(test_roundtrip_model())

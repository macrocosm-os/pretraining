import asyncio
import os
from model.data import Model, ModelId
from model.storage.model_store import ModelStore
from transformers import AutoModel, DistilBertModel, DistilBertConfig


class HuggingFaceModelStore(ModelStore):
    """Hugging Face based implementation for storing and retrieving a model."""

    async def store_model(self, model: Model):
        """Stores a trained model in Hugging Face."""
        token = os.getenv("HF_ACCESS_TOKEN")

        # PreTrainedModel.save_pretrained only saves locally
        model.pt_model.push_to_hub(
            repo_id=model.id.path + "/" + model.id.name, token=token
        )

    # TODO actually make this asynchronous with threadpools etc.
    async def retrieve_model(self, model_id: ModelId) -> Model:
        """Retrieves a trained model from Hugging Face or the local Hugging Face cache."""
        # Transformers library can pick up a model based on the hugging face path (username/model) + rev.
        # Will default to the local hf cache first, then fall back on going to the hub.
        model = AutoModel.from_pretrained(
            model_id.path + "/" + model_id.name,
            revision=model_id.rev,
        )

        return Model(id=model_id, pt_model=model)


async def test_roundtrip_model():
    """Verifies that the HuggingFaceModelStore can roundtrip a model in hugging face."""
    hf_name = os.getenv("HF_NAME")
    model_id = ModelId(
        path=hf_name,
        name="TestModel",
        hash="TestHash1",
        rev="main",
    )

    pt_model = DistilBertModel(
        config=DistilBertConfig(
            vocab_size=256, n_layers=2, n_heads=4, dim=100, hidden_dim=400
        )
    )

    model = Model(id=model_id, pt_model=pt_model)
    hf_model_store = HuggingFaceModelStore()

    # Store the model in hf.
    await hf_model_store.store_model(model=model)

    # Retrieve the model from hf.
    retrieved_model = await hf_model_store.retrieve_model(model_id=model_id)

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
        rev="6f9ad47",
    )

    hf_model_store = HuggingFaceModelStore()

    # Retrieve the model from hf (first run) or cache.
    model = await hf_model_store.retrieve_model(model_id=model_id)

    print(f"Finished retrieving the model with id: {model.id}")


if __name__ == "__main__":
    asyncio.run(test_retrieve_model())
    # asyncio.run(test_roundtrip_model())

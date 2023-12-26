import abc
from model.data import Model, ModelId


class ModelStore(abc.ABC):
    """An abstract base class for storing and retrieving a pre trained model."""

    @abc.abstractmethod
    async def store_model(self, hotkey: str, model: Model) -> ModelId:
        """Stores a trained model in the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    async def retrieve_model(self, hotkey: str, pt_model_id: ModelId) -> Model:
        """Retrieves a trained model from the appropriate location based on implementation."""
        pass

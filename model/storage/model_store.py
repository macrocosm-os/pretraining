import abc
from model.data import Model, ModelId


class ModelStore(abc.ABC):
    """An abstract base class for storing and retrieving a pre trained model."""

    @abc.abstractmethod
    async def store_model(self, uid: int, model: Model):
        """Stores a trained model in the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    async def retrieve_model(self, uid: int, pt_model_id: ModelId) -> Model:
        """Retrieves a trained model from the appropriate location based on implementation."""
        pass

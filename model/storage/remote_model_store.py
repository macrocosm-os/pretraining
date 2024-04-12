import abc
import sys
from model.data import Model, ModelId


class RemoteModelStore(abc.ABC):
    """An abstract base class for storing and retrieving a pre trained model."""

    @abc.abstractmethod
    async def upload_model(self, model: Model, use_bf16_and_flash: bool) -> ModelId:
        """Uploads a trained model in the appropriate location based on implementation."""
        pass

    @abc.abstractmethod
    async def download_model(
        self,
        model_id: ModelId,
        local_path: str,
        use_bf16_and_flash: bool,
        model_size_limit: int = sys.maxsize,
    ) -> Model:
        """Retrieves a trained model from the appropriate location and stores at the given path."""
        pass

from typing import Any, Dict
from pydantic import BaseModel, ConfigDict, Field
from transformers import PreTrainedModel


class ModelId(BaseModel):
    """Uniquely identifies a trained model"""

    # Makes the object "Immutable" once created.
    model_config = ConfigDict(frozen=True)

    path: str = Field(
        description="Path to where this model can be found. ex. a huggingface.io repo."
    )
    name: str = Field(description="Name of the model.")
    # TODO: Consider only using commit hashes for revision and remove need for content hash.
    rev: str = Field(description="Revision of the model.")
    hash: str = Field(description="Hash of the trained model.")


class Model(BaseModel):
    """Represents a pre trained foundation model."""

    class Config:
        arbitrary_types_allowed = True

    id: ModelId = Field(description="Identifier for this model.")
    # PreTrainedModel.base_model returns torch.nn.Module if needed.
    pt_model: PreTrainedModel = Field(description="Pre trained model.")


class ModelMetadata(BaseModel):
    id: ModelId = Field(description="Identifier for this trained model.")
    # TODO consider making this a timestamp by converting the block.
    block: int = Field(
        description="Block on which this model was claimed on the chain."
    )

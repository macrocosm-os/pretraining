from typing import Any, Dict, Type
from transformers import PreTrainedModel
from pydantic import BaseModel, ConfigDict, Field


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

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return f"{self.path}:{self.name}:{self.rev}:{self.hash}"

    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["ModelId"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":")
        return cls(
            path=tokens[0],
            name=tokens[1],
            rev=tokens[2],
            hash=tokens[3],
        )


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

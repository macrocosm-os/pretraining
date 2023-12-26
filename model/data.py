from typing import Any, Dict, Optional, Type
from transformers import PreTrainedModel
from pydantic import BaseModel, Field, PositiveInt


class ModelId(BaseModel):
    """Uniquely identifies a trained model"""

    # Makes the object "Immutable" once created.
    class Config:
        frozen = True
        extra = "forbid"

    # TODO add pydantic validations on underlying fields.
    path: str = Field(
        description="Path to where this model can be found. ex. a huggingface.io repo."
    )
    name: str = Field(description="Name of the model.")
    commit: Optional[str] = Field(
        description="Commit of the model. May be empty if not yet commited."
    )
    hash: str = Field(description="Hash of the trained model.")

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return f"{self.path}:{self.name}:{self.commit}:{self.hash}"

    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["ModelId"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":")
        return cls(
            path=tokens[0],
            name=tokens[1],
            commit=tokens[2],
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
    block: PositiveInt = Field(
        description="Block on which this model was claimed on the chain."
    )

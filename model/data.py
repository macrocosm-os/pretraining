import dataclasses
from enum import IntEnum
from typing import Any, ClassVar, Dict, Optional, Type
from transformers import PreTrainedModel
from pydantic import BaseModel, Field, PositiveInt

# The maximum bytes for metadata on the chain.
MAX_METADATA_BYTES = 128
# The length, in bytes, of a git commit hash.
GIT_COMMIT_LENGTH = 40
# The length, in bytes, of a base64 encoded sha256 hash.
SHA256_BASE_64_LENGTH = 44


class ModelId(BaseModel):
    """Uniquely identifies a trained model"""

    # Makes the object "Immutable" once created.
    class Config:
        frozen = True
        extra = "forbid"

    MAX_REPO_ID_LENGTH: ClassVar[int] = (
        MAX_METADATA_BYTES - GIT_COMMIT_LENGTH - SHA256_BASE_64_LENGTH - 3  # separators
    )

    namespace: str = Field(
        description="Namespace where the model can be found. ex. Hugging Face username/org."
    )
    name: str = Field(description="Name of the model.")

    # When handling a model locally the commit and hash are not necessary.
    # Commit must be filled when trying to download from a remote store.
    commit: Optional[str] = Field(
        description="Commit of the model. May be empty if not yet committed."
    )
    # Hash is filled automatically when uploading to or downloading from a remote store.
    hash: Optional[str] = Field(description="Hash of the trained model.")

    def to_compressed_str(self) -> str:
        """Returns a compressed string representation."""
        return f"{self.namespace}:{self.name}:{self.commit}:{self.hash}"

    @classmethod
    def from_compressed_str(cls, cs: str) -> Type["ModelId"]:
        """Returns an instance of this class from a compressed string representation"""
        tokens = cs.split(":")
        return cls(
            namespace=tokens[0],
            name=tokens[1],
            commit=tokens[2] if tokens[2] != "None" else None,
            hash=tokens[3] if tokens[3] != "None" else None,
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


class TokenizerIdentifier(IntEnum):
    """Identifiers the tokenizer to use. This may mean different tokenizers or different implementations."""

    DISTILGPT_2 = 1
    GPT3_5_TURBO_16K = 2


@dataclasses.dataclass()
class ModelCriteria:
    """Collection of criteria that are relevant for interacting with models in the subnet."""

    # Sequence length used for inference.
    sequence_length: int

    # Whether bfloat16 and flash attention optimizations should be used.
    optimized: bool

    # Maximum model size in bytes.
    max_model_bytes: int

    # Maximum model parameter size.
    max_model_parameters: int

    # Allowed model types.
    allowed_model_types: dict[type]

    # Tokenizer to use.
    tokenizer_identifier: TokenizerIdentifier

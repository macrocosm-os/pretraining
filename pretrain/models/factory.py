from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

import constants
from competitions.data import CompetitionId

from taoverse.model.tts.auto import AutoModelForTTS

class ModelFactory:
    @staticmethod
    def get_model(
        model_dir: str,
        competition_id: str
    ) -> Union[nn.Module, PreTrainedModel]:
        """Loads the model object for a given competition.

        Every competition requires specific model class types.
        This methods uses the appropriate loader for a given competition.
        """

        # Get current model parameters
        model_constraints = constants.MODEL_CONSTRAINTS_BY_COMPETITION_ID.get(
            competition_id, None
        )

        if model_constraints is None:
            raise RuntimeError(
                f"Could not find current competition for id: {competition_id}"
            )

        match competition_id:
            case CompetitionId.TTS_V0:
                return AutoModelForTTS.from_pretrained(
                    pretrained_model_name_or_path=model_dir,
                )

            # By default we use CausalLM HF loaders
            case _:
                return AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_dir,
                    local_files_only=True,
                    use_safetensors=True,
                    **model_constraints.kwargs,
                )

import unittest
from model.utils import get_model_criteria
from constants import BLOCK_7B, ALLOWED_MODEL_TYPES_1, ALLOWED_MODEL_TYPES_2
from model.data import ModelCriteria, TokenizerIdentifier


class TestModelUtils(unittest.TestCase):
    MODEL_CRITERIA_772M = ModelCriteria(
        sequence_length=1024,
        optimized=False,
        max_model_bytes=5 * 1024 * 1024 * 1024,
        max_model_parameters=772_000_000,
        allowed_model_types=ALLOWED_MODEL_TYPES_1,
        tokenizer_identifier=TokenizerIdentifier.DISTILGPT_2,
    )
    MODEL_CRITERIA_7B = ModelCriteria(
        sequence_length=8192,
        optimized=True,
        max_model_bytes=15 * 1024 * 1024 * 1024,
        max_model_parameters=6_900_000_000,
        allowed_model_types=ALLOWED_MODEL_TYPES_2,
        tokenizer_identifier=TokenizerIdentifier.GPT3_5_TURBO_16K,
    )
    model_criteria_cases = [
        (2_405_920, MODEL_CRITERIA_772M),
        (2_605_920, MODEL_CRITERIA_772M),
        (BLOCK_7B - 1, MODEL_CRITERIA_772M),
        (BLOCK_7B, MODEL_CRITERIA_7B),
        (BLOCK_7B + 1, MODEL_CRITERIA_7B),
    ]

    def test_get_model_criteria(self):
        for block, expected_criteria in self.model_criteria_cases:
            with self.subTest(block=block, expected_criteria=expected_criteria):
                assert get_model_criteria(block) == expected_criteria


if __name__ == "__main__":
    unittest.main()

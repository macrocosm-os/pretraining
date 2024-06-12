import unittest
from model.utils import get_model_criteria, get_hash_of_two_strings
from constants import BLOCK_7B, BLOCK_FW_EDU_SCORE_2, ALLOWED_MODEL_TYPES_1, ALLOWED_MODEL_TYPES_2, DATASET_1, DATASET_2
from model.data import ModelCriteria, TokenizerIdentifier


class TestModelUtils(unittest.TestCase):
    MODEL_CRITERIA_186M = ModelCriteria(
        sequence_length=1024,
        optimized=False,
        max_model_bytes=5 * 1024 * 1024 * 1024,
        max_model_parameters=186_000_000,
        allowed_model_types=ALLOWED_MODEL_TYPES_1,
        tokenizer_identifier=TokenizerIdentifier.DISTILGPT_2,
        evaluation_dataset=DATASET_1,
    )
    MODEL_CRITERIA_772M = ModelCriteria(
        sequence_length=1024,
        optimized=False,
        max_model_bytes=5 * 1024 * 1024 * 1024,
        max_model_parameters=772_000_000,
        allowed_model_types=ALLOWED_MODEL_TYPES_1,
        tokenizer_identifier=TokenizerIdentifier.DISTILGPT_2,
        evaluation_dataset=DATASET_1,
    )
    MODEL_CRITERIA_7B = ModelCriteria(
        sequence_length=4096,
        optimized=True,
        max_model_bytes=15 * 1024 * 1024 * 1024,
        max_model_parameters=6_900_000_000,
        allowed_model_types=ALLOWED_MODEL_TYPES_2,
        tokenizer_identifier=TokenizerIdentifier.GPT_4_TIKTOKEN,
        evaluation_dataset=DATASET_1,
    )
    MODEL_CRITERIA_FINEWEBEDU2 = ModelCriteria(
        sequence_length=4096,
        optimized=True,
        max_model_bytes=15 * 1024 * 1024 * 1024,
        max_model_parameters=6_900_000_000,
        allowed_model_types=ALLOWED_MODEL_TYPES_2,
        tokenizer_identifier=TokenizerIdentifier.GPT_4_TIKTOKEN,
        evaluation_dataset=DATASET_2,
    )
    
    model_criteria_cases = [
        (0, MODEL_CRITERIA_186M),
        (2_405_919, MODEL_CRITERIA_186M),
        (2_405_920, MODEL_CRITERIA_772M),
        (2_605_920, MODEL_CRITERIA_772M),
        (BLOCK_7B - 1, MODEL_CRITERIA_772M),
        (BLOCK_7B, MODEL_CRITERIA_7B),
        (BLOCK_7B + 1, MODEL_CRITERIA_7B),
        (BLOCK_FW_EDU_SCORE_2 - 1, MODEL_CRITERIA_7B),
        (BLOCK_FW_EDU_SCORE_2, MODEL_CRITERIA_FINEWEBEDU2),
        (BLOCK_FW_EDU_SCORE_2 + 1, MODEL_CRITERIA_FINEWEBEDU2),
    ]

    def test_get_model_criteria(self):
        for block, expected_criteria in self.model_criteria_cases:
            with self.subTest(block=block, expected_criteria=expected_criteria):
                assert get_model_criteria(block) == expected_criteria

    def test_get_hash_of_two_strings(self):
        string1 = "hello"
        string2 = "world"

        result = get_hash_of_two_strings(string1, string2)

        self.assertEqual(result, "k2oYXKqiZrucvpgengXLeM1zKwsygOuURBK7b4+PB68=")


if __name__ == "__main__":
    unittest.main()

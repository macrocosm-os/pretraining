import constants
from model.data import ModelCriteria


def get_model_criteria(block: int) -> ModelCriteria:
    """Returns the model criteria at block."""
    criteria = None
    for b, crit in constants.MODEL_CRITERIA_BY_BLOCK:
        if block >= b:
            criteria = crit
    assert criteria is not None, f"No model criteria found for block {block}"
    return criteria

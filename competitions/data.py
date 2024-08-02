from enum import IntEnum


class CompetitionId(IntEnum):
    """Unique identifiers for each competition."""

    B7_MODEL = 0

    M772_MODEL = 1

    B3_MODEL = 2

    B7_MODEL_LOWER_EPSILON = 3

    # Overwrite the default __repr__, which doesn't work with
    # bt.logging for some unknown reason.
    def __repr__(self) -> str:
        return f"{self.value}"

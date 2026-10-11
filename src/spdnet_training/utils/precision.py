"""Derive Lightning's `trainer.precision` from the model's requested dtype."""

_DTYPE_TO_PRECISION = {"float32": 32, "float": 32, "float64": 64, "double": 64}


def derive_precision_from_dtype(model_dtype: str | None) -> int | None:
    """
    Map a model dtype string (as found in `model.dtype` config) to the
    matching Lightning `trainer.precision` value.

    Returns None when `model_dtype` is unset or not one of the known
    aliases, meaning the caller should leave `trainer.precision` untouched.
    """
    if model_dtype is None:
        return None
    return _DTYPE_TO_PRECISION.get(str(model_dtype))

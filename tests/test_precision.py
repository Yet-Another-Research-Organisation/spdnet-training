from spdnet_training.utils.precision import derive_precision_from_dtype


def test_float64_aliases_map_to_64():
    assert derive_precision_from_dtype("float64") == 64
    assert derive_precision_from_dtype("double") == 64


def test_float32_aliases_map_to_32():
    assert derive_precision_from_dtype("float32") == 32
    assert derive_precision_from_dtype("float") == 32


def test_none_leaves_precision_untouched():
    assert derive_precision_from_dtype(None) is None


def test_unknown_dtype_leaves_precision_untouched():
    assert derive_precision_from_dtype("bfloat16") is None

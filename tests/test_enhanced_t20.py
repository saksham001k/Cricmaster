import math

from cricmaster.models.enhanced_t20 import shrunk_rate


def test_shrunk_rate_uses_neutral_prior() -> None:
    assert math.isclose(shrunk_rate(0, 0), 0.5)
    assert math.isclose(shrunk_rate(5, 10), 0.5)
    assert 0.5 < shrunk_rate(10, 10) < 1.0


def test_shrunk_rate_handles_invalid_values() -> None:
    assert math.isnan(shrunk_rate(None, None))

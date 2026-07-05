"""Tests for the factorial function defined in factorial.py."""

import pytest
from factorial import factorial

@pytest.mark.parametrize("input_val, expected", [
    (0, 1),
    (1, 1),
    (2, 2),
    (3, 6),
    (5, 120),
    (7, 5040),
])
def test_factorial_correctness(input_val, expected):
    assert factorial(input_val) == expected

def test_factorial_negative():
    with pytest.raises(ValueError):
        factorial(-1)

def test_factorial_non_integer():
    with pytest.raises(ValueError):
        factorial(3.5)

"""Unit tests for math scoring function."""

import pytest
from llamafactory.train.grpo.reward.math import math_score


class TestMathScore:
    """Tests for math_score function."""

    def test_boxed_exact_match(self):
        response = "The answer is \\boxed{42}."
        ground_truth = "42"
        assert math_score(response, ground_truth, extract_mode="boxed") == 1.0

    def test_boxed_no_match(self):
        response = "The answer is \\boxed{43}."
        ground_truth = "42"
        assert math_score(response, ground_truth, extract_mode="boxed") == 0.0

    def test_boxed_no_boxed_present(self):
        response = "The answer is 42."
        ground_truth = "42"
        assert math_score(response, ground_truth, extract_mode="boxed") == 0.0

    def test_hash_exact_match(self):
        response = "Calculate: 3 + 4 = #### 7"
        ground_truth = "7"
        assert math_score(response, ground_truth, extract_mode="hash") == 1.0

    def test_hash_no_match(self):
        response = "Calculate: 3 + 4 = #### 8"
        ground_truth = "7"
        assert math_score(response, ground_truth, extract_mode="hash") == 0.0

    def test_last_number_match(self):
        response = "Therefore, the answer is 3.14."
        ground_truth = "3.14"
        assert math_score(response, ground_truth, extract_mode="last_number") == 1.0

    def test_normalized_match(self):
        """Test that normalization (commas, spaces, dots) works correctly."""
        response = "The answer is \\boxed{1,234.56}."
        ground_truth = "1234.56"
        assert math_score(response, ground_truth, extract_mode="boxed") == 1.0

    def test_negative_number(self):
        response = "The result is \\boxed{-5}."
        ground_truth = "-5"
        assert math_score(response, ground_truth, extract_mode="boxed") == 1.0

    def test_fraction_normalization(self):
        response = "The answer is \\boxed{1/2}."
        ground_truth = "1/2"
        assert math_score(response, ground_truth, extract_mode="boxed") == 1.0

    def test_empty_response(self):
        response = ""
        ground_truth = "42"
        assert math_score(response, ground_truth, extract_mode="boxed") == 0.0

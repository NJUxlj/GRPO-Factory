"""Unit tests for string match scoring function."""

from llamafactory.train.grpo.reward.string_match import string_match_score


class TestStringMatch:
    """Tests for string_match_score function."""

    def test_exact_match_strict(self):
        assert string_match_score("hello world", "hello world", strict=True) == 1.0

    def test_different_strings_strict(self):
        assert string_match_score("hello world", "hello moon", strict=True) == 0.0

    def test_extra_spaces_strict(self):
        # Strict mode collapses whitespace
        assert string_match_score("hello  world", "hello world", strict=True) == 1.0

    def test_case_difference_relaxed(self):
        # Relaxed mode lowercases
        assert string_match_score("Hello World", "hello world", strict=False) == 1.0

    def test_punctuation_relaxed(self):
        # Relaxed mode strips punctuation
        assert string_match_score("Hello, World!", "hello world", strict=False) == 1.0

    def test_spaces_relaxed(self):
        # Relaxed mode removes all whitespace
        assert string_match_score(" hello  world ", "helloworld", strict=False) == 1.0

    def test_empty_strings(self):
        assert string_match_score("", "", strict=True) == 1.0
        assert string_match_score("", "", strict=False) == 1.0

    def test_whitespace_only(self):
        assert string_match_score("   ", "", strict=True) == 1.0
        assert string_match_score("   ", "", strict=False) == 1.0

"""Unit tests for multiple choice scoring function."""

from llamafactory.train.grpo.reward.multiple_choice import multiple_choice_score


class TestMultipleChoice:
    """Tests for multiple_choice_score function."""

    def test_boxed_format(self):
        response = "The answer is \\boxed{A}."
        ground_truth = "A"
        assert multiple_choice_score(response, ground_truth) == 1.0

    def test_answer_format(self):
        response = "answer: B"
        ground_truth = "B"
        assert multiple_choice_score(response, ground_truth) == 1.0

    def test_chinese_answer_format(self):
        response = "答案：C"
        ground_truth = "C"
        # Use a pattern that also matches Chinese "答案"
        pattern = r"(?i)\\boxed\{\s*([A-D])\s*\}|(?:answer|答案)\s*[:：]?\s*([A-D])"
        assert multiple_choice_score(response, ground_truth, pattern=pattern) == 1.0

    def test_case_insensitive(self):
        response = "\\boxed{a}"
        ground_truth = "A"
        assert multiple_choice_score(response, ground_truth) == 1.0

    def test_wrong_answer(self):
        response = "\\boxed{D}"
        ground_truth = "A"
        assert multiple_choice_score(response, ground_truth) == 0.0

    def test_no_choice_found(self):
        response = "I don't know the answer."
        ground_truth = "A"
        assert multiple_choice_score(response, ground_truth) == 0.0

    def test_multiple_choices_take_first(self):
        """The default pattern matches \\boxed{...} first."""
        response = "\\boxed{A} or \\boxed{B}"
        ground_truth = "A"
        assert multiple_choice_score(response, ground_truth) == 1.0

    def test_ground_truth_whitespace(self):
        response = "\\boxed{C}"
        ground_truth = "  C  "
        assert multiple_choice_score(response, ground_truth) == 1.0

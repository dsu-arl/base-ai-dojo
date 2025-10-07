"""Unit tests for the check_answer function."""

import unittest
from base_test import BaseTestValidator


class TestAnswerCheck(BaseTestValidator):
    """Unit tests for the check_answer method of the Validator class.

    This class tests the validation logic of the check_answer method, which verifies
    that the user's value for the 'n_estimators' variables matches the correct answer.
    Inherits from BaseTestValidator to reuse common setup and utility methods for
    creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests check_answer with valid variable declaration with the correct value for
        the 'n_estimators' variable.

        Verifies that check_answer returns (True, "") when the user's 'n_estimators'
        variable has a value of 140.
        """
        content = "n_estimators = 140"
        validator = self.create_validator(content)
        is_correct, msg = validator.check_answer()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")

    def test_missing_variable(self):
        """Tests check_answer where the 'n_estimators' variable is not defined.

        Verifies that check_answer returns (False, "Make sure you have a variable called
        'n_estimators'") when the user's code does not contain the 'n_estimators'
        variable.
        """
        content = ""
        validator = self.create_validator(content)
        is_correct, msg = validator.check_answer()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Make sure you have a variable called 'n_estimators'")

    def test_incorrect_variable_type(self):
        """Tests check_answer when the value of the 'n_estimators' variable is the wrong
        data type.

        Verifies that check_answer returns (False, "Make sure you're assigning an
        integer value to 'n_estimators'") when the 'n_estimators' variable's value has a
        data type of anything other than an integer.
        """
        cases = [
            {"desc": "String variable", "content": "n_estimators = 'test'"},
            {"desc": "Float variable", "content": "n_estimators = 123.45"},
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.check_answer()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg, "Make sure you're assigning an integer value to 'n_estimators'"
                )

    def test_answer_too_low(self):
        """Tests check_answer when the value of the 'n_estimators' variable is lower
        than the correct answer.

        Verifies that check_answer returns (False, "Incorrect (HINT: Higher
        'n_estimators' value)") when the value of the 'n_estimators' variable is lower
        than the correct answer.
        """
        content = "n_estimators = 100"
        validator = self.create_validator(content)
        is_correct, msg = validator.check_answer()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Incorrect (HINT: Higher 'n_estimators' value)")

    def test_answer_too_high(self):
        """Tests check_answer when the value of the 'n_estimators' variable is higher
        than the correct answer.

        Verifies that check_answer returns (False, "Incorrect (HINT: Lower
        'n_estimators' value)") when the value of the 'n_estimators' variable is higher
        than the correct answer.
        """
        content = "n_estimators = 150"
        validator = self.create_validator(content)
        is_correct, msg = validator.check_answer()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Incorrect (HINT: Lower 'n_estimators' value)")


if __name__ == "__main__":
    unittest.main()

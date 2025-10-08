"""Unit tests for the step_5_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep5Check(BaseTestValidator):
    """Unit tests for the step_5_check method of the Validator class.

    This class tests the validation logic of the step_5_check method, which verifies
    that the user's code correctly calculates the accuracy of the model using the
    predictions the true outputs. Inherits from BaseTestValidator to reuse common setup
    and utility methods for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_5_check with valid code that correctly calculates the model's
        accuracy.

        Verifies that step_5_check returns (True, "") when the code calls the
        accuracy_score function, supplying parameters either positional or using
        keywords, and storing the output in a variable. Uses subTests to check multiple
        correct cases.
        """
        cases = [
            {
                "desc": "Passing parameters using args",
                "content": "accuracy = accuracy_score(y_test, y_pred)",
            },
            {
                "desc": "Passing parameters using kwargs",
                "content": "accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.user_vars.y_test = "y_test"
                validator.user_vars.y_pred = "y_pred"
                is_correct, msg = validator.step_5_check()
                self.assertEqual(validator.user_vars.accuracy, "accuracy")
                self.assertTrue(is_correct)
                self.assertEqual(msg, "")

    def test_missing_function_call(self):
        """Tests step_5_check when accuracy_score is not called in the code.

        Verifies that step_5_check returns (False, "accuracy_score() isn't called")
        when the code is empty, indicating the function was not called.
        """
        content = ""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "accuracy_score() isn't called")

    def test_function_called_more_than_once(self):
        """Tests step_5_check when accuracy_score is called multiple times.

        Verifies that step_5_check returns (False, "accuracy_score() should only be
        called once") when the code contains multiple calls to accuracy_score,
        indicating a violation of the requirement to call the function exactly once.
        """
        content = """
accuracy = accuracy_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "accuracy_score() should only be called once")

    def test_output_not_assigned(self):
        """Tests step_5_check when output of accuracy_score is not stored in a
        variable.

        Verifies that step_5_check returns (False, "Make sure you store the output of
        accuracy_score() in a variable") when accuracy_score is called with the output
        not stored in a variable.
        """
        content = "accuracy_score(y_test, y_pred)"
        validator = self.create_validator(content)
        is_correct, msg = validator.step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Make sure you store the output of accuracy_score() in a variable"
        )

    def test_incorrect_args(self):
        """Tests step_5_check when accuracy_score is called with incorrect positional or
        keyword arguments.

        Verifies that step_5_check returns (False, "Incorrect parameters for
        accuracy_score(), are you correctly passing the test output and model output to
        it?") when accuracy_score is called with incorrect positional or keyword
        arguments. Uses subTests to check multiple argument error cases.
        """
        cases = [
            {
                "desc": "Empty kwargs with incorrect args",
                "content": "accuracy = accuracy_score(y_train, y_test)",
            },
            {
                "desc": "Empty args with incorrect kwargs",
                "content": "accuracy = accuracy_score(y_true=y_train, y_pred=y_test)",
            },
            {
                "desc": "Empty args and empty kwargs",
                "content": "accuracy = accuracy_score()",
            },
            {
                "desc": "Incorrect number of args",
                "content": "accuracy = accuracy_score(y_test)",
            },
            {
                "desc": "Incorrect number of kwargs",
                "content": "accuracy = accuracy_score(y_true=y_test)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.user_vars.y_test = "y_test"
                validator.user_vars.y_pred = "y_pred"
                is_correct, msg = validator.step_5_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "Incorrect parameters for accuracy_score(), are you correctly "
                    "passing the test output and model output to it?",
                )


if __name__ == "__main__":
    unittest.main()

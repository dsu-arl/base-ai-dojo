"""Unit tests for the step_5_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep5Check(BaseTestValidator):
    """Unit tests for the step_5_check method of the Validator class.

    This class tests the validation logic of the step_5_check method, which verifies
    that the user's code correctly calculates the mean squared error on the model's
    predictions and the true outputs. Inherits from BaseTestValidator to reuse common
    setup and utility methods for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_5_check with valid code that correctly calculates the mean squared
        error.

        Verifies that step_5_check returns (True, "") when the code calls the
        mean_squared_error function, supplying parameters either positional or using
        keywords, and storing the output in a variable. Uses subTests to check multiple
        correct cases.
        """
        cases = [
            {
                "desc": "Passing parameters using args",
                "content": "mse = mean_squared_error(y_test, y_pred)",
            },
            {
                "desc": "Passing parameters using kwargs",
                "content": "mse = mean_squared_error(y_true=y_test, y_pred=y_pred)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.user_vars.y_test = "y_test"
                validator.user_vars.y_pred = "y_pred"
                is_correct, msg = validator.step_5_check()
                self.assertEqual(validator.user_vars.mse, "mse")
                self.assertTrue(is_correct)
                self.assertEqual(msg, "")

    def test_missing_function_call(self):
        """Tests step_5_check when mean_squared_error is not called in the code.

        Verifies that step_5_check returns (False, "mean_squared_error() isn't called")
        when the code is empty, indicating the function was not called.
        """
        content = ""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "mean_squared_error() isn't called")

    def test_function_called_more_than_once(self):
        """Tests step_5_check when mean_squared_error is called multiple times.

        Verifies that step_5_check returns (False, "mean_squared_error() should only be
        called once") when the code contains multiple calls to mean_squared_error,
        indicating a violation of the requirement to call the function exactly once.
        """
        content = """
mse = mean_squared_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "mean_squared_error() should only be called once")

    def test_output_not_assigned(self):
        """Tests step_5_check when output of mean_squared_error is not stored in a
        variable.

        Verifies that step_5_check returns (False, "Make sure you store the output of
        mean_squared_error() in a variable") when mean_squared_error is called with the
        output not stored in a variable.
        """
        content = "mean_squared_error(y_test, y_pred)"
        validator = self.create_validator(content)
        is_correct, msg = validator.step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Make sure you store the output of mean_squared_error() in a variable"
        )

    def test_incorrect_args(self):
        """Tests step_5_check when mean_squared_error is called with incorrect
        positional or keyword arguments.

        Verifies that step_5_check returns (False, "Incorrect parameters for
        mean_squared_error(), are you correctly passing the test output and model output
        to it?") when mean_squared_error is called with incorrect positional or keyword
        arguments. Uses subTests to check multiple argument error cases.
        """
        cases = [
            {
                "desc": "Empty kwargs with incorrect args",
                "content": "mse = mean_squared_error(y_train, y_test)",
            },
            {
                "desc": "Empty args with incorrect kwargs",
                "content": "mse = mean_squared_error(y_true=y_train, y_pred=y_test)",
            },
            {
                "desc": "Empty args and empty kwargs",
                "content": "mse = mean_squared_error()",
            },
            {
                "desc": "Incorrect number of args",
                "content": "mse = mean_squared_error(y_true=y_test)",
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
                    "Incorrect parameters for mean_squared_error(), are you correctly "
                    + "passing the test output and model output to it?",
                )


if __name__ == "__main__":
    unittest.main()

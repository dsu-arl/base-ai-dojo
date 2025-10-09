"""Unit tests for the step_7_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep7Check(BaseTestValidator):
    """Unit tests for the step_7_check method of the Validator class.

    This class tests the validation logic of the step_7_check method, which verifies
    that the user's code correctly rounds the model's predictions. Inherits from
    BaseTestValidator to reuse common setup and utility methods for creating Validator
    instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_7_check with valid code that correctly rounds the predictions.

        Verifies that step_7_check returns (True, "") when the .round function is called
        on the predictions with the output being stored in either the same variable or a
        new one, and the output variable name is stored in
        validator.user_vars.predictions.
        """
        content = "rounded_preds = predictions.round()"
        validator = self.create_validator(content)
        validator.user_vars.predictions = "predictions"
        is_correct, msg = validator.step_7_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")
        self.assertEqual(validator.user_vars.predictions, "rounded_preds")

    def test_round_not_called(self):
        """Tests step_7_check when .round is not called in the code.

        Verifies that step_7_check returns (False, "predictions.round() isn't called")
        when the code doesn't contain a call to .round.
        """
        content = ""
        validator = self.create_validator(content)
        validator.user_vars.predictions = "predictions"
        is_correct, msg = validator.step_7_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "predictions.round() isn't called")

    def test_round_called_more_than_once(self):
        """Tests step_7_check when .round is called more than once.

        Verifies that step_7_check returns (False, "predictions.round() should only be
        called once") when the code contains multiple calls to predictions.round.
        """
        content = """
predictions1 = predictions.round()
predictions2 = predictions.round()
"""
        validator = self.create_validator(content)
        validator.user_vars.predictions = "predictions"
        is_correct, msg = validator.step_7_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "predictions.round() should only be called once")

    def test_output_not_assigned(self):
        """Tests step_7_check when the output of .round is not assigned to a variable.

        Verifies that step_7_check returns (False, "Make sure you store the output of
        predictions.round() in a variable") when the output of predictions.round is not
        stored in a variable.
        """
        content = "predictions.round()"
        validator = self.create_validator(content)
        validator.user_vars.predictions = "predictions"
        is_correct, msg = validator.step_7_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Make sure you store the output of predictions.round() in a variable"
        )

    def test_incorrect_args_kwargs(self):
        """Tests step_7_check when .round has any arguments passed to it.

        Verifies that step_7_check returns (False, "predictions.round() shouldn't have
        any parameters passed to it for this challenge") when predictions.round has any
        arguments passed to it.
        """
        content = "predictions = predictions.round(23, test_arg='test')"
        validator = self.create_validator(content)
        validator.user_vars.predictions = "predictions"
        is_correct, msg = validator.step_7_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg,
            "predictions.round() shouldn't have any parameters "
            "passed to it for this challenge",
        )


if __name__ == "__main__":
    unittest.main()

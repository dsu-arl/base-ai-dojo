"""Unit tests for the step_9_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep9Check(BaseTestValidator):
    """Unit tests for the step_9_check method of the Validator class.

    This class tests the validation logic of the step_9_check method, which verifies
    that the user's code correctly prints the rounded predictions in the required
    format. Inherits from BaseTestValidator to reuse common setup and utility methods
    for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_9_check with valid code that correctly prints out the predictions.

        Verifies that step_9_check returns (True, "") when the code correctly prints out
        the rounded predictions in the required format using .numpy().
        """
        content = "print(predictions.numpy())"
        validator = self.create_validator(content)
        validator.data_user_vars.predictions = "predictions"
        is_correct, msg = validator.step_9_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")

    def test_print_not_called(self):
        """Tests step_9_check when the print statement is not called.

        Verifies that step_9_check returns (False, "print isn't called") when the print
        statement to print the rounded predictions is missing.
        """
        content = ""
        validator = self.create_validator(content)
        validator.data_user_vars.predictions = "predictions"
        is_correct, msg = validator.step_9_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() isn't called")

    def test_print_output_stored(self):
        """Tests step_9_check when the output of print is stored in a variable.

        Verifies that step_9_check returns (False, "print() shouldn't be assigned to
        any variables") when the code calls print but stores its output in a variable.
        """
        content = "test = print(predictions.numpy())"
        validator = self.create_validator(content)
        validator.data_user_vars.predictions = "predictions"
        is_correct, msg = validator.step_9_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() shouldn't be assigned to any variables")

    def test_incorrect_args(self):
        """Tests step_9_check when the value to be printed out is incorrect.

        Verifies that step_9_check returns (False, "Incorrect data to be printed out or
        not in the required format for the challenge") when print is called but the data
        passed to it is incorrect or the data passed to it doesn't call .numpy(). Uses
        subTests to check multiple argument error cases.
        """
        cases = [
            {"desc": "Incorrect data", "content": "print(y.numpy())"},
            {"desc": "No .numpy() call", "content": "print(predictions)"},
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.data_user_vars.predictions = "predictions"
                is_correct, msg = validator.step_9_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "Incorrect data to be printed out or not in the "
                    "required format for the challenge",
                )


if __name__ == "__main__":
    unittest.main()

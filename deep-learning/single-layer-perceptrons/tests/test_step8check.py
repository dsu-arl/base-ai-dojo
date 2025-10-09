"""Unit tests for the step_8_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep8Check(BaseTestValidator):
    """Unit tests for the step_8_check method of the Validator class.

    This class tests the validation logic of the step_8_check method, which verifies
    that the user's code correctly prints out the model's rounded predictions. Inherits
    from BaseTestValidator to reuse common setup and utility methods for creating
    Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_8_check with valid code that correctly prints out the rounded
        predictions.

        Verifies that step_8_check returns (True, "") when the print function is called
        on the predictions.
        """
        content = "print(predictions)"
        validator = self.create_validator(content)
        validator.user_vars.predictions = "predictions"
        is_correct, msg = validator.step_8_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")

    def test_print_not_called(self):
        """Tests step_8_check when print is not called in the code.

        Verifies that step_8_check returns (False, "print() isn't called") when the code
        doesn't contain a call to print.
        """
        content = ""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() isn't called")

    def test_print_called_more_than_once(self):
        """Tests step_8_check when print is called more than once.

        Verifies that step_8_check returns (False, "print() should only be called once")
        when the code contains multiple calls to print.
        """
        content = """
print()
print()
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() should only be called once")

    def test_output_assigned(self):
        """Tests step_8_check when the output of print is stored in a variable.

        Verifies that step_8_check returns (False, "print() shouldn't be assigned to any
        variables") when the output of print is stored in a variable.
        """
        content = "test = print()"
        validator = self.create_validator(content)
        validator.user_vars.predictions = "predictions"
        is_correct, msg = validator.step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() shouldn't be assigned to any variables")

    def test_incorrect_print_value(self):
        """Tests step_8_check when the wrong values are printed out.

        Verifies that step_8_check returns (False, "Incorrect parameters passed to
        print(), are you passing your rounded predictions to the print() function?")
        when the rounded predictions aren't the values being printed.
        """
        content = "print(X)"
        validator = self.create_validator(content)
        validator.user_vars.predictions = "predictions"
        is_correct, msg = validator.step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg,
            "Incorrect parameters passed to print(), are you passing "
            "your rounded predictions to the print() function?",
        )


if __name__ == "__main__":
    unittest.main()

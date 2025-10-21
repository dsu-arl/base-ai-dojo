"""Unit tests for the step_6_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep6Check(BaseTestValidator):
    """Unit tests for the step_6_check method of the Validator class.

    This class tests the validation logic of the step_6_check method, which verifies
    that the user's code correctly prints the accuracy calculated in the
    previous step. Inherits from BaseTestValidator to reuse common setup and utility
    methods for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_6_check with valid code that correctly prints out the accuracy.

        Verifies that step_6_check returns (True, "") when the code uses the print
        function to output the accuracy.
        """
        content = "print(accuracy)"
        validator = self.create_validator(content)
        validator.user_vars.accuracy = "accuracy"
        is_correct, msg = validator.step_6_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")

    def test_missing_function_call(self):
        """Tests step_6_check when print is not called in the code.

        Verifies that step_6_check returns (False, "print() isn't called") when the code
        is empty, indicating the function was not called.
        """
        content = ""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() isn't called")

    def test_function_called_more_than_once(self):
        """Tests step_6_check when print is called multiple times.

        Verifies that step_6_check returns (False, "print() should only be called once")
        when the code contains multiple calls to print, indicating a violation of the
        requirement to call the function exactly once.
        """
        content = """
print(accuracy)
print(accuracy)
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() should only be called once")

    def test_output_assigned(self):
        """Tests step_6_check when the output of print is attempted to be stored in a
        variable.

        Verifies that step_6_check returns (False, "print() shouldn't be assigned to any
        variables") when print is called with the output is stored in a variable.
        """
        content = "test = print(accuracy)"
        validator = self.create_validator(content)
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() shouldn't be assigned to any variables")

    def test_incorrect_args(self):
        """Tests step_6_check when print is called with incorrect arguments.

        Verifies that step_6_check returns (False, "Are you printing out the correct
        variable for the accuracy?") when print is called with the incorrect variable
        name for the variable containing the calculated accuracy. Uses subTests to
        check multiple argument error cases.
        """
        cases = [
            {"desc": "No variable given to print", "content": "print()"},
            {
                "desc": "Incorrect variable name given",
                "content": "print(incorrect_accuracy)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.user_vars.accuracy = "accuracy"
                is_correct, msg = validator.step_6_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg, "Are you printing out the correct variable for the accuracy?"
                )

    def test_incorrect_kwargs(self):
        """Tests step_6_check when print is called with incorrect keyword arguments.

        Verifies that step_6_check returns (False, "You don't need any keyword arguments
        for this print statement") when print is called with any keyword arguments.
        """
        content = "print(accuracy, test=accuracy)"
        validator = self.create_validator(content)
        validator.user_vars.accuracy = "accuracy"
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "You don't need any keyword arguments for this print statement"
        )


if __name__ == "__main__":
    unittest.main()

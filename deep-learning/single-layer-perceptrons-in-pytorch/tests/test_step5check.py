"""Unit tests for the step_5_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep5Check(BaseTestValidator):
    """Unit tests for the step_5_check method of the Validator class.

    This class tests the validation logic of the step_5_check method, which verifies
    that the user's code correctly initializes the perceptron model using the created
    Perceptron class. Inherits from BaseTestValidator to reuse common setup and utility
    methods for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_5_check with valid code that initializes the perceptron model.

        Verifies that step_5_check returns (True, "") when the code creates an instance
        of the Perceptron class and storing the variable name in
        validator.model_user_vars.model.
        """
        content = "model = Perceptron()"
        validator = self.create_validator(content)
        is_correct, msg = validator.step_5_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")
        self.assertEqual(validator.model_user_vars.model, "model")

    def test_perceptron_not_called(self):
        """Tests step_5_check when the Perceptron class is not called.

        Verifies that step_5_check returns (False, "Perceptron() isn't called") when the
        code doesn't call the Perceptron class.
        """
        content = ""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Perceptron() isn't called")

    def test_class_called_more_than_once(self):
        """Tests step_5_check when Perceptron is called more than once.

        Verifies that step_5_check returns (False, "Perceptron() should only be called
        once") when the code contains more than one call to Perceptron().
        """
        content = """
model = Perceptron()
model = Perceptron()
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Perceptron() should only be called once")

    def test_output_not_assigned(self):
        """Tests step_5_check when output of Perceptron() is not stored in any variables

        Verifies that step_5_check returns (False, "Make sure you store the output of
        Perceptron() in a variable") when the code calls Perceptron() with no output
        variables specified.
        """
        content = "Perceptron()"
        validator = self.create_validator(content)
        is_correct, msg = validator.step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Make sure you store the output of Perceptron() in a variable"
        )

    def test_perceptron_call_has_args(self):
        """Tests step_5_check when Perceptron is called with arguments.

        Verifies that step_5_check returns (False, "You shouldn't be passing any
        parameters to Perceptron() for this challenge") when Perceptron is called with
        arguments. Uses subTests to check multiple argument error cases.
        """
        cases = [
            {"desc": "Has args", "content": "model = Perceptron(23)"},
            {"desc": "Has kwargs", "content": "model = Perceptron(test=23)"},
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_5_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "You shouldn't be passing any parameters to Perceptron() "
                    "for this challenge",
                )


if __name__ == "__main__":
    unittest.main()

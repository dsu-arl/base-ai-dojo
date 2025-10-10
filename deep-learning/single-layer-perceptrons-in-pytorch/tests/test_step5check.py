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


if __name__ == "__main__":
    unittest.main()

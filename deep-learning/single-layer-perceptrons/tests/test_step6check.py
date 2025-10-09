"""Unit tests for the step_6_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep6Check(BaseTestValidator):
    """Unit tests for the step_6_check method of the Validator class.

    This class tests the validation logic of the step_6_check method, which verifies
    that the user's code correctly uses the trained perceptron to generate predictions.
    Inherits from BaseTestValidator to reuse common setup and utility methods for
    creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_6_check with valid code that correctly makes predictions using the
        trained model.

        Verifies that step_6_check returns (True, "") when the .predict function is
        called on the model with 'X' as the input data, output is stored in a
        variable, and the output variable name is stored in
        validator.user_vars.predictions.
        """
        content = "predictions = model.predict(X)"
        validator = self.create_validator(content)
        validator.user_vars.x = "X"
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_6_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")
        self.assertEqual(validator.user_vars.predictions, "predictions")

    def test_predict_not_called(self):
        """Tests step_6_check when .predict is not called in the code.

        Verifies that step_6_check returns (False, "model.predict() isn't called") when
        the code doesn't contain a call to .predict.
        """
        content = ""
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.predict() isn't called")

    def test_predict_called_more_than_once(self):
        """Tests step_6_check when .predict is called more than once.

        Verifies that step_6_check returns (False, "model.predict() should only be
        called once") when the code contains multiple calls to model.predict.
        """
        content = """
predictions1 = model.predict(X)
predictions2 = model.predict(X)
"""
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.predict() should only be called once")

    def test_output_not_assigned(self):
        """Tests step_6_check when the output of .predict is not assigned to a variable.

        Verifies that step_6_check returns (False, "Make sure you store the output of
        model.predict() in a variable") when the output of model.predict is not stored
        in a variable.
        """
        content = "model.predict(X)"
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Make sure you store the output of model.predict() in a variable"
        )

    def test_incorrect_args(self):
        """Tests step_6_check when .predict has incorrect positional arguments.

        Verifies that step_6_check returns (False, "model.predict() should take a single
        argument X") when model.predict doesn't have the input data X passed to it.
        """
        content = "predictions = model.predict(X, y)"
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        validator.user_vars.x = "X"
        validator.user_vars.y = "y"
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.predict() should take a single argument X")

    def test_incorrect_kwargs(self):
        """Tests step_6_check when .predict has keyword arguments specified.

        Verifies that step_6_check returns (False, "model.predict() should not have any
        keyword arguments for this challenge") when model.predict contains keyword
        arguments.
        """
        content = "predictions = model.predict(X, test_arg='test')"
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        validator.user_vars.x = "X"
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg,
            "model.predict() should not have any keyword arguments for this challenge",
        )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the step_5_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep5Check(BaseTestValidator):
    """Unit tests for the step_5_check method of the Validator class.

    This class tests the validation logic of the step_5_check method, which verifies
    that the user's code correctly trains the perceptron with the correct parameters.
    Inherits from BaseTestValidator to reuse common setup and utility methods for
    creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_5_check with valid code that correctly trains the model.

        Verifies that step_5_check returns (True, "") when the .fit function is called
        on the model with 100 epochs and '1' for the verbose parameter.
        """
        content = "model.fit(X, y, epochs=100, verbose=1)"
        validator = self.create_validator(content)
        validator.user_vars.x = "X"
        validator.user_vars.y = "y"
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_5_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")

    def test_fit_not_called(self):
        """Tests step_5_check when model.fit is not called in the code.

        Verifies that step_5_check returns (False, "model.compile() isn't called") when
        the code doesn't contain a call to model.fit.
        """
        content = ""
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.fit() isn't called")

    def test_fit_called_more_than_once(self):
        """Tests step_5_check when model.fit is called more than once.

        Verifies that step_5_check returns (True, "mode.fit() should only be called
        once") when the code contains multiple calls to model.fit.
        """
        content = """
model.fit(X, y, epochs=100, verbose=1)
model.fit(X, y, epochs=100, verbose=1)
"""
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.fit() should only be called once")

    def test_fit_output_assigned_to_variable(self):
        """Tests step_5_check when the output of model.fit is assigned to a variable.

        Verifies that step_5_check returns (False, "model.fit() shouldn't be assigned to
        any variables") when the output of model.fit is stored in a variable.
        """
        content = "test = model.fit(X, y, epochs=100, verbose=1)"
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.fit() shouldn't be assigned to any variables")

    def test_incorrect_fit_args(self):
        """Tests step_5_check when model.fit has incorrect positional arguments.

        Verifies that step_5_check returns (False, "Missing or incorrect parameters, are
        you passing your dataset and labels to model.fit()?") when the dataset or labels
        aren't correctly passed to model.fit. Uses subTests to check multiple error
        cases.
        """
        cases = [
            {
                "desc": "Missing X parameter",
                "content": "model.fit([], y, epochs=100, verbose=1)",
            },
            {
                "desc": "Missing y parameter",
                "content": "model.fit(X, [], epochs=100, verbose=1)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.user_vars.x = "X"
                validator.user_vars.y = "y"
                validator.user_vars.model = "model"
                is_correct, msg = validator.step_5_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "Missing or incorrect parameters, are you passing your "
                    "dataset and labels to model.fit()?",
                )

    def test_incorrect_fit_kwargs(self):
        """Tests step_5_check when model.fit has incorrect keyword arguments.

        Verifies that step_5_check returns (False, "Missing or incorrect parameters, are
        you training for 100 epochs and outputting the training output?") when the value
        for parameters 'epochs' or 'verbose' do not match what was mentioned in the
        instructions. Uses subTests to check multiple error cases.
        """
        cases = [
            {
                "desc": "Missing epochs parameter",
                "content": "model.fit(X, y, verbose=1)",
            },
            {
                "desc": "Incorrect epochs parameter",
                "content": "model.fit(X, y, epochs=50, verbose=1)",
            },
            {
                "desc": "Missing verbose parameter",
                "content": "model.fit(X, y, epochs=100)",
            },
            {
                "desc": "Incorrect verbose parameter",
                "content": "model.fit(X, y, epochs=100, verbose=0)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.user_vars.x = "X"
                validator.user_vars.y = "y"
                validator.user_vars.model = "model"
                is_correct, msg = validator.step_5_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "Missing or incorrect parameters, are you training for "
                    "100 epochs and outputting the training output?",
                )


if __name__ == "__main__":
    unittest.main()

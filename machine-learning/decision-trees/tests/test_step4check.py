"""Unit tests for the step_4_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep4Check(BaseTestValidator):
    """Unit tests for the step_4_check method of the Validator class.

    This class tests the validation logic of the step_4_check method, which verifies
    that the user's code correctly uses the trained decision tree model to make
    predictions on the test dataset and stores in an output variable. Inherits from
    BaseTestValidator to reuse common setup and utility methods for creating Validator
    instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_4_check with valid code that correctly uses the trained decision
        tree model to make predictions on the test dataset.

        Verifies that step_4_check returns (True, "") when the script contains
        "y_pred = model.predict(X_test)" or "y_pred = model.predict(X=X_test)". Also
        verifies that 'y_pred' is correctly stored in validator.user_vars.y_pred. Uses
        subTests to check multiple correct cases.
        """
        cases = [
            {
                "desc": "Passing parameters using args",
                "content": "y_pred = model.predict(X_test)",
            },
            {
                "desc": "Passing parameters using kwargs",
                "content": "y_pred = model.predict(X=X_test)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.user_vars.model = "model"
                validator.user_vars.x_test = "X_test"
                is_correct, msg = validator.step_4_check()
                self.assertEqual(validator.user_vars.y_pred, "y_pred")
                self.assertTrue(is_correct)
                self.assertEqual(msg, "")

    def test_missing_function_call(self):
        """Tests step_4_check when model.predict is not called in the code.

        Verifies that step_4_check returns (False, "model.predict() isn't called") when
        the code is empty, indicating the function was not called.
        """
        content = ""
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        validator.user_vars.x_test = "X_test"
        is_correct, msg = validator.step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.predict() isn't called")

    def test_function_called_more_than_once(self):
        """Tests step_4_check when model.predict is called multiple times.

        Verifies that step_4_check returns (False, "model.predict() should only be
        called once") when the code contains multiple calls to model.predict, indicating
        a violation of the requirement to call the function exactly once.
        """
        content = """
y_pred = model.predict(X_test)
y_pred = model.predict(X_test)
"""
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        validator.user_vars.x_test = "X_test"
        is_correct, msg = validator.step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.predict() should only be called once")

    def test_output_not_assigned(self):
        """Tests step_4_check when output of model.predict is not stored in a variable.

        Verifies that step_4_check returns (False, "Make sure you store the output of
        model.predict() in a variable") when model.predict is called with the output not
        stored in a variable.
        """
        content = "model.predict(X_test)"
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        validator.user_vars.x_test = "X_test"
        is_correct, msg = validator.step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Make sure you store the output of model.predict() in a variable"
        )

    def test_incorrect_args(self):
        """Tests step_4_check when model.predict is called with incorrect positional or
        keyword arguments.

        Verifies that step_4_check returns (False, "Incorrect parameters for
        model.predict(), are you correctly passing the test data to it?") when
        model.predict is called with incorrect positional or keyword arguments. Uses
        subTests to check multiple argument error cases.
        """
        cases = [
            {
                "desc": "Empty kwargs with incorrect args",
                "content": "y_pred = model.predict(y_train)",
            },
            {
                "desc": "Empty args with incorrect kwargs",
                "content": "y_pred = model.predict(X=y_train)",
            },
            {
                "desc": "Non-empty args and non-empty kwargs",
                "content": "y_pred = model.predict(X_test, X=X_test)",
            },
            {
                "desc": "Empty args and empty kwargs",
                "content": "y_pred = model.predict()",
            },
            {
                "desc": "Incorrect number of args",
                "content": "y_pred = model.predict(X_train, y_train)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.user_vars.model = "model"
                validator.user_vars.x_test = "X_test"
                is_correct, msg = validator.step_4_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "Incorrect parameters for model.predict(), are you correctly "
                    "passing the test data to it?",
                )


if __name__ == "__main__":
    unittest.main()

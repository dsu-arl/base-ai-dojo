"Unit tests for the step_3_check function."

import unittest
from base_test import BaseTestValidator


class TestStep3Check(BaseTestValidator):
    """Unit tests for the step_3_check method of the Validator class.

    This class tests the validation logic of the step_3_check method, which verifies
    that the user's code correctly trains a decision tree model on the dataset. Inherits
    from BaseTestValidator to reuse common setup and utility methods for creating
    Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_3_check with valid code that correctly trains the decision tree
        model.

        Verifies that step_3_check returns (True, "") when the script contains
        "model.fit(X_train, y_train)" or "model.fit(X=X_train, y=y_train)". Uses
        subTests to check multiple correct cases.
        """
        cases = [
            {
                "desc": "Passing parameters using args",
                "content": "model.fit(X_train, y_train)",
            },
            {
                "desc": "Passing parameters using kwargs",
                "content": "model.fit(X=X_train, y=y_train)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.user_vars.model = "model"
                validator.user_vars.x_train = "X_train"
                validator.user_vars.y_train = "y_train"
                is_correct, msg = validator.step_3_check()
                self.assertTrue(is_correct)
                self.assertEqual(msg, "")

    def test_missing_function_call(self):
        """Tests step_3_check when model.fit is not called in the code.

        Verifies that step_3_check returns (False, "model.fit() isn't called") when the
        code is empty, indicating the function was not called.
        """
        content = ""
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.fit() isn't called")

    def test_function_called_more_than_once(self):
        """Tests step_3_check when model.fit is called multiple times.

        Verifies that step_3_check returns (False, "model.fit() should only be called
        once") when the code contains multiple calls to model.fit, indicating a
        violation of the requirement to call the function exactly once.
        """
        content = """
model.fit(X_train, y_train)
model.fit(X_train, y_train)
"""
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.fit() should only be called once")

    def test_output_assigned(self):
        """Tests step_3_check when output of model.fit is stored in variable.

        Verifies that step_3_check returns (False, "model.fit() shouldn't be assigned to
        any variables") when model.fit is called with the output assigned to a variable.
        """
        content = "test = model.fit(X_train, y_train)"
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.fit() shouldn't be assigned to any variables")

    def test_incorrect_args(self):
        """Tests step_3_check when model.fit is called with incorrect positional or
        keyword arguments.

        Verifies that step_3_check returns (False, "Incorrect parameters for
        model.fit(), are you correctly passing the training data to it?") when model.fit
        is called with incorrect positional or keyword arguments. Uses subTests to check
        multiple argument error cases.
        """
        cases = [
            {
                "desc": "Empty kwargs with incorrect args",
                "content": "model.fit(X_test, y_test)",
            },
            {
                "desc": "Empty args with incorrect kwargs",
                "content": "model.fit(X=X_test, y=y_test)",
            },
            {
                "desc": "Non-empty args and non-empty kwargs",
                "content": "model.fit(X_train, y_train, X=X_train, y=y_train)",
            },
            {"desc": "Empty args and empty kwargs", "content": "model.fit()"},
            {"desc": "Incorrect number of args", "content": "model.fit(X_train)"},
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.user_vars.model = "model"
                is_correct, msg = validator.step_3_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "Incorrect parameters for model.fit(), are you correctly passing "
                    + "the training data to it?",
                )


if __name__ == "__main__":
    unittest.main()

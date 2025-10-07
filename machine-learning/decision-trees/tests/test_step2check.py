"""Unit tests for the step_2_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep2Check(BaseTestValidator):
    """Unit tests for the step_2_check method of the Validator class.

    This class tests the validation logic of the step_2_check method, which verifies
    that the user's code correctly creates a decision tree model and stores it in a
    variable. Inherits from BaseTestValidator to reuse common setup and utility methods
    for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_2_check with valid code that calls DecisionTreeClassifier
        correctly.

        Verifies that step_2_check returns (True, "") when the script contains
        "model = DecisionTreeClassifier()", and that the variable 'model' is stored
        correctly in validator.user_vars.model.
        """
        content = "model = DecisionTreeClassifier()"
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertEqual(validator.user_vars.model, "model")
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")

    def test_missing_function_call(self):
        """Tests step_2_check when DecisionTreeClassifier is not called in the code.

        Verifies that step_2_check returns (False, "DecisionTreeClassifier() isn't
        called") when the code is empty, indicating the function was not invoked.
        """
        content = ""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "DecisionTreeClassifier() isn't called")

    def test_function_called_more_than_once(self):
        """Tests step_2_check when DecisionTreeClassifier is called multiple times.

        Verifies that step_2_check returns (False, "DecisionTreeClassifier() should only
        be called once") when the code contains multiple calls to
        DecisionTreeClassifier, indicating a violation of the requirement to call the
        function exactly once.
        """
        content = """
model = DecisionTreeClassifier()
model = DecisionTreeClassifier()
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "DecisionTreeClassifier() should only be called once")

    def test_output_not_assigned(self):
        """Tests step_2_check when output of DecisionTreeClassifier() is not stored in
        any variables.

        Verifies that step_2_check returns (False, "Make sure you store the output of
        DecisionTreeClassifier() in a variable") when the code calls
        DecisionTreeClassifier with not output variables specified.
        """
        content = "DecisionTreeClassifier()"
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg,
            "Make sure you store the output of DecisionTreeClassifier() in a variable",
        )

    def test_function_has_args(self):
        """Tests step_2_check when DecisionTreeClassifier is called with arguments.

        Verifies that step_2_check returns (False, "You shouldn't be passing any
        parameters to DecisionTreeClassifier() for this challenge") when
        DecisionTreeClassifier is called with arguments. Uses subTests to check multiple
        argument error cases.
        """
        cases = [
            {"desc": "Has args", "content": "model = DecisionTreeClassifier(23)"},
            {
                "desc": "Has kwargs",
                "content": "model = DecisionTreeClassifier(test=23)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_2_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "You shouldn't be passing any parameters to "
                    "DecisionTreeClassifier() for this challenge",
                )


if __name__ == "__main__":
    unittest.main()

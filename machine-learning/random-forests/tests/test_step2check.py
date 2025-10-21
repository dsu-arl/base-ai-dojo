"""Unit tests for the step_2_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep2Check(BaseTestValidator):
    """Unit tests for the step_2_check method of the Validator class.

    This class tests the validation logic of the step_2_check method, which verifies
    that the user's code correctly creates a random forest model and stores it in a
    variable. Inherits from BaseTestValidator to reuse common setup and utility methods
    for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_2_check with valid code that calls RandomForestClassifier
        correctly.

        Verifies that step_2_check returns (True, "") when the script contains
        "model = RandomForestClassifier()", and that the variable 'model' is stored
        correctly in validator.user_vars.model.
        """
        content = "model = RandomForestClassifier()"
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertEqual(validator.user_vars.model, "model")
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")

    def test_missing_function_call(self):
        """Tests step_2_check when RandomForestClassifier is not called in the code.

        Verifies that step_2_check returns (False, "RandomForestClassifier() isn't
        called") when the code is empty, indicating the function was not invoked.
        """
        content = ""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "RandomForestClassifier() isn't called")

    def test_function_called_more_than_once(self):
        """Tests step_2_check when RandomForestClassifier is called multiple times.

        Verifies that step_2_check returns (False, "RandomForestClassifier() should only
        be called once") when the code contains multiple calls to
        RandomForestClassifier, indicating a violation of the requirement to call the
        function exactly once.
        """
        content = """
model = RandomForestClassifier()
model = RandomForestClassifier()
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "RandomForestClassifier() should only be called once")

    def test_output_not_assigned(self):
        """Tests step_2_check when output of RandomForestClassifier() is not stored in
        any variables.

        Verifies that step_2_check returns (False, "Make sure you store the output of
        RandomForestClassifier() in a variable") when the code calls
        RandomForestClassifier with not output variables specified.
        """
        content = "RandomForestClassifier()"
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg,
            "Make sure you store the output of RandomForestClassifier() in a variable",
        )

    def test_function_has_args(self):
        """Tests step_2_check when RandomForestClassifier is called with arguments.

        Verifies that step_2_check returns (False, "You shouldn't be passing any
        parameters to RandomForestClassifier() for this challenge") when
        RandomForestClassifier is called with arguments. Uses subTests to check multiple
        argument error cases.
        """
        cases = [
            {"desc": "Has args", "content": "model = RandomForestClassifier(23)"},
            {
                "desc": "Has kwargs",
                "content": "model = RandomForestClassifier(test=23)",
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
                    "RandomForestClassifier() for this challenge",
                )


if __name__ == "__main__":
    unittest.main()

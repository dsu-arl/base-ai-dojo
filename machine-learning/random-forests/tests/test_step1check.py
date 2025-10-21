"""Unit tests for the step_1_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep1Check(BaseTestValidator):
    """Unit tests for the step_1_check method of the Validator class.

    This class tests the validation logic of the step_1_check method, which verifies
    that the user's code correctly splits the dataset into training and testing sets and
    stores the result in the correct variables. Inherits from BaseTestValidator to reuse
    common setup and utility methods for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_1_check with valid code that calls train_test_split correctly.

        Verifies that step_1_check returns (True, "") when the script contains
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
        random_state=23)", and that the variables 'X_train', 'X_test', 'y_train',
        and 'y_test' are stored correctly in validator.user_vars.
        """
        content = (
            "X_train, X_test, y_train, y_test = "
            + "train_test_split(X, y, test_size=0.3, random_state=23)"
        )
        validator = self.create_validator(content)
        is_correct, msg = validator.step_1_check()
        self.assertEqual(validator.user_vars.x_train, "X_train")
        self.assertEqual(validator.user_vars.x_test, "X_test")
        self.assertEqual(validator.user_vars.y_train, "y_train")
        self.assertEqual(validator.user_vars.y_test, "y_test")
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")

    def test_missing_function_call(self):
        """Tests step_1_check when train_test_split is not called in the code.

        Verifies that step_1_check returns (False, "train_test_split() isn't called")
        when the code is empty, indicating the function was not invoked.
        """
        content = ""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "train_test_split() isn't called")

    def test_function_called_more_than_once(self):
        """Tests step_1_check when train_test_split is called multiple times.

        Verifies that step_1_check returns (False, "train_test_split() should only be
        called once") when the code contains multiple calls to train_test_split,
        indicating a violation of the requirement to call the function exactly once.
        """
        content = """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "train_test_split() should only be called once")

    def test_function_output_not_stored(self):
        """Tests step_1_check when output of train_test_split is not stored in any
        variables.

        Verifies that step_1_check returns (False, "Make sure you store the output of
        train_test_split() in a variable") when the code calls train_test_split with no
        output variables specified.
        """
        content = "train_test_split(X, y, test_size=0.3, random_state=23)"
        validator = self.create_validator(content)
        is_correct, msg = validator.step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg,
            "Make sure you store the output of train_test_split() in a variable",
        )

    def test_incorrect_number_output_variables(self):
        """Tests step_1_check when train_test_split is called with the incorrect number
        of output variables.

        Verifies that step_1_check returns (False, "train_test_split() should be
        unpacked into 4 variables") when the code calls train_test_split with the
        incorrect number of output variables. Uses subTests to check multiple output
        variable number error cases.
        """
        cases = [
            {
                "desc": "3 output variables",
                "content": "X_train, X_test, y_train = "
                "train_test_split(X, y, test_size=0.3, random_state=23)",
            },
            {
                "desc": "5 output variables",
                "content": "X_train, X_test, y_train, y_test, test = "
                "train_test_split(X, y, test_size=0.3, random_state=23)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_1_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg, "train_test_split() should be unpacked into 4 variables"
                )

    def test_incorrect_args(self):
        """Tests step_1_check when train_test_split is called with the incorrect
        positional arguments.

        Verifies that step_1_check returns (False, "You need to pass X and y to
        train_test_split() so it knows what data to split") for function calls with
        incorrect positional arguments. Uses subTests to check multiple argument
        error cases.
        """
        cases = [
            {
                "desc": "Incorrect X",
                "content": "X_train, X_test, y_train, y_test = "
                "train_test_split([], y, test_size=0.3, random_state=23)",
            },
            {
                "desc": "Incorrect y",
                "content": "X_train, X_test, y_train, y_test = "
                "train_test_split(X, test_size=0.3, random_state=23)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_1_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "You need to pass X and y to train_test_split() "
                    "so it knows what data to split",
                )

    def test_incorrect_kwargs(self):
        """Tests step_1_check when train_test_split is called with incorrect keyword
        arguments.

        Verifies that step_1_check returns (False, "train_test_split() parameters don't
        match what's expected, did you give 'test_size' and 'random_state' the correct
        values from the instructions?") for function calls with incorrect keyword
        arguments.
        """
        cases = [
            {
                "desc": "Incorrect test size",
                "content": "X_train, X_test, y_train, y_test = "
                + "train_test_split(X, y, test_size=0.8, random_state=23)",
            },
            {
                "desc": "Incorrect random state",
                "content": "X_train, X_test, y_train, y_test = "
                "train_test_split(X, y, test_size=0.3, random_state=32)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_1_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "train_test_split() parameters don't match what's expected, "
                    "did you give 'test_size' and 'random_state' the correct "
                    "values from the instructions?",
                )


if __name__ == "__main__":
    unittest.main()

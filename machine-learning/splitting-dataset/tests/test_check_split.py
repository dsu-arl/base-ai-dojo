"""Unit tests for the check_split() function."""

import unittest
from base_test import BaseTestValidator


class TestCheckSplit(BaseTestValidator):
    """Unit tests for the check_split method of the Validator class.

    This class tests the validation logic of the check_split method, which verifies that
    the user's code correctly calls train_test_split and stores the result in a
    variable. Inherits from BaseTestValidator to reuse common setup and utility methods
    for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities,
            including the create_validator method.
    """

    def test_success(self):
        """Tests check_split with a valid script that calls train_test_split correctly.

        Verifies that check_y returns (True, "") when the script contains
        'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)',
        and that the variable names 'X_train', 'X_test', 'y_train', and 'y_test' are
        stored correctly in validator.user_vars.
        """
        script_content = (
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)"
        )
        validator = self.create_validator(script_content)
        validator.user_vars.x = "X"
        validator.user_vars.y = "y"
        is_correct, msg = validator.check_split()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")
        self.assertEqual(validator.user_vars.x_train, "X_train")
        self.assertEqual(validator.user_vars.x_test, "X_test")
        self.assertEqual(validator.user_vars.y_train, "y_train")
        self.assertEqual(validator.user_vars.y_test, "y_test")

    def test_split_not_called(self):
        """Tests check_split when train_test_split is not called in the script.

        Verifies that check_split returns (False, "train_test_split() isn't called")
        when the script is empty, indicating the function was not invoked.
        """
        script_content = ""
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_split()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "train_test_split() isn't called")

    def test_split_called_more_than_once(self):
        """Tests check_split when train_test_split is called multiple times.

        Verifies that check_split returns (False, "train_test_split() should only be
        called once") when the script contains multiple calls to train_test_split,
        indicating a violation of the requirement to call the function exactly once.
        """
        script_content = """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
"""
        validator = self.create_validator(script_content)
        validator.user_vars.x = "X"
        validator.user_vars.y = "y"
        is_correct, msg = validator.check_split()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "train_test_split() should only be called once")

    def test_output_not_stored(self):
        """Tests check_split when train_test_split is called without storing the result.

        Verifies that check_split returns a failure with the message "Make sure you're
        storing the output of train_test_split() in variables for your train and test
        datasets (HINT: It's 4 variables)" when the script calls train_test_split
        without assigning it to a variable.
        """
        script_content = "train_test_split(X, y, test_size=0.3)"
        validator = self.create_validator(script_content)
        validator.user_vars.x = "X"
        validator.user_vars.y = "y"
        is_correct, msg = validator.check_split()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg,
            "Make sure you're storing the output of train_test_split() in variables "
            + "for your train and test datasets (HINT: It's 4 variables)",
        )

    def test_incorrect_number_output_variables(self):
        """Tests check_split when train_test_split is called with the incorrect number
        of output variables.

        Verifies that check_split returns a failure with the message "Incorrect number
        of output variables." when the script calls train_test_split without assigning
        its output to exactly 4 variables. Uses subTests to check multiple output
        variable number error cases.
        """
        cases = [
            {
                "desc": "Only 3 output variables",
                "script_content": (
                    "X_train, X_test, y_train = train_test_split(X, y, test_size=0.3)"
                ),
            },
            {
                "desc": "Only 2 output variables",
                "script_content": (
                    "X_train, X_test = train_test_split(X, y, test_size=0.3)"
                ),
            },
            {
                "desc": "Only 1 output variable",
                "script_content": "X_train = train_test_split(X, y, test_size=0.3)",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["script_content"])
                validator.user_vars.x = "X"
                validator.user_vars.y = "y"
                is_correct, msg = validator.check_split()
                self.assertFalse(is_correct)
                self.assertEqual(msg, "Incorrect number of output variables.")

    def test_incorrect_args(self):
        """Tests check_split when train_test_split is called with the incorrect
        positional arguments.

        Verifies that check_split returns a failure with the message "Did you provide
        the correct variables for X and y?" for function calls with incorrect positional
        arguments. Uses subTests to check multiple argument error cases.
        """
        cases = [
            {
                "desc": "Incorrect X",
                "script_content": (
                    "X_train, X_test, y_train, y_test = "
                    + "train_test_split([], y, test_size=0.3)"
                ),
            },
            {
                "desc": "Incorrect y",
                "script_content": (
                    "X_train, X_test, y_train, y_test = "
                    + "train_test_split(X, test_size=0.3)"
                ),
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["script_content"])
                validator.user_vars.x = "X"
                validator.user_vars.y = "y"
                is_correct, msg = validator.check_split()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg, "Did you provide the correct variables for X and y?"
                )

    def test_incorrect_kwargs(self):
        """Tests check_split when train_test_split is called with incorrect keyword
        arguments.

        Verifies that check_split returns a failure with the message "Did you correctly
        set the 'test_size' parameter so that the training dataset is 70% of the
        original dataset?" for function calls with incorrect keyword arguments.
        """
        script_content = (
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)"
        )
        validator = self.create_validator(script_content)
        validator.user_vars.x = "X"
        validator.user_vars.y = "y"
        is_correct, msg = validator.check_split()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg,
            "Did you correctly set the 'test_size' parameter so that the training "
            + "dataset is 70% of the original dataset?",
        )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the check_y() function."""

import unittest
from base_test import BaseTestValidator


class TestCheckY(BaseTestValidator):
    """Unit tests for the check_y method of the Validator class.

    This class tests the validation logic of the check_y method, which verifies
    that the user's code correctly calls np.random.randint(0, 2, size=500) and stores
    the result in a variable. Inherits from BaseTestValidator to reuse common setup
    and utility methods for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities,
            including the create_validator method.
    """

    def test_success(self):
        """Tests check_y with a valid script that calls np.random.randint correctly.

        Verifies that check_y returns (True, "") when the script contains
        'y = np.random.randint(0, 2, size=500)', and that the variable 'y' is stored
        correctly in validator.user_vars.y.
        """
        script_content = "y = np.random.randint(0, 2, size=500)"
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_y()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")
        self.assertEqual(validator.user_vars.y, "y")

    def test_rand_not_called(self):
        """Tests check_y when np.random.randint is not called in the script.

        Verifies that check_y returns (False, "np.random.randint() isn't called")
        when the script is empty, indicating the function was not invoked.
        """
        script_content = ""
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_y()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "np.random.randint() isn't called")

    def test_rand_called_more_than_once(self):
        """Tests check_y when np.random.randint is called multiple times.

        Verifies that check_y returns (False, "np.random.randint() should only be called
        once") when the script contains multiple calls to np.random.randint, indicating
        a violation of the requirement to call the function exactly once.
        """
        script_content = """
y = np.random.randint(0, 2, size=500)
y = np.random.randint(0, 2, size=500)
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_y()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "np.random.randint() should only be called once")

    def test_output_not_assigned_to_variable(self):
        """Tests check_y when np.random.randint is called without storing the result.

        Verifies that check_y returns a failure with the message
        "Make sure you're storing the output of np.random.randint() in a variable"
        when the script calls np.random.randint without assigning it to a variable.
        """
        script_content = "np.random.randint(0, 5, size=500)"
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_y()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg,
            "Make sure you're storing the output of np.random.randint() in a variable",
        )

    def test_incorrect_args(self):
        """Tests check_y with incorrect arguments to np.random.rand.

        Verifies that check_y returns a failure with the message
        "Did you pass the correct values in the instructions to np.random.randint()?"
        for scripts with incorrect arguments. Uses subTests to check multiple argument
        error cases.
        """
        cases = [
            {
                "desc": "Incorrect low parameter value",
                "script_content": "y = np.random.randint(1, 2, size=500)",
            },
            {
                "desc": "Incorrect high parameter value",
                "script_content": "y = np.random.randint(0, 5, size=500)",
            },
            {
                "desc": "Incorrect size parameter value",
                "script_content": "y = np.random.randint(0, 2, size=100)",
            },
        ]
        error_msg = (
            "Did you pass the correct values in the instructions "
            + "to np.random.randint()?"
        )
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["script_content"])
                is_correct, msg = validator.check_y()
                self.assertFalse(is_correct)
                self.assertEqual(msg, error_msg)


if __name__ == "__main__":
    unittest.main()

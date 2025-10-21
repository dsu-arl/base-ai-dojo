"""Unit tests for the check_x() function."""

import unittest
from base_test import BaseTestValidator


class TestCheckX(BaseTestValidator):
    """Unit tests for the check_x method of the Validator class.

    This class tests the validation logic of the check_x method, which verifies
    that the user's code correctly calls np.random.rand(500, 7) and stores the
    result in a variable. Inherits from BaseTestValidator to reuse
    common setup and utility methods for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities,
            including the create_validator method.
    """

    def test_success(self):
        """Tests check_x with a valid script that calls np.random.rand correctly.

        Verifies that check_x returns (True, "") when the script contains
        'x = np.random.rand(500, 7)', and that the variable 'x' is stored
        correctly in validator.user_vars.x.
        """
        script_content = "X = np.random.rand(500, 7)"
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_x()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")
        self.assertEqual(validator.user_vars.x, "X")

    def test_rand_not_called(self):
        """Tests check_x when np.random.rand is not called in the script.

        Verifies that check_x returns (False, "np.random.rand() isn't called")
        when the script is empty, indicating the function was not invoked.
        """
        script_content = ""
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_x()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "np.random.rand() isn't called")

    def test_rand_called_more_than_once(self):
        """Tests check_x when np.random.rand is called multiple times.

        Verifies that check_x returns (False, "np.random.rand() should only be called
        once") when the script contains multiple calls to np.random.rand, indicating a
        violation of the requirement to call the function exactly once.
        """
        script_content = """
X = np.random.rand(500, 7)
X = np.random.rand(500, 7)
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_x()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "np.random.rand() should only be called once")

    def test_output_not_assigned_to_variable(self):
        """Tests check_x when np.random.rand is called without storing the result.

        Verifies that check_x returns a failure with the message
        "Make sure you're storing the output of np.random.rand() in a variable"
        when the script calls np.random.rand(500, 7) without assigning it to a variable.
        """
        script_content = "np.random.rand(500, 7)"
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_x()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Make sure you're storing the output of np.random.rand() in a variable"
        )

    def test_incorrect_args(self):
        """Tests check_x with incorrect arguments to np.random.rand.

        Verifies that check_x returns a failure with the message
        "Did you pass the correct values in the instructions to np.random.rand()?" for
        scripts with incorrect arguments (e.g., wrong number of samples or features).
        Uses subTests to check multiple argument error cases.
        """
        cases = [
            {
                "desc": "Incorrect number of samples",
                "script_content": "X = np.random.rand(5, 3)",
            },
            {
                "desc": "Incorrect number of features",
                "script_content": "X = np.random.rand(500, 30)",
            },
        ]
        error_msg = (
            "Did you pass the correct values in the instructions to np.random.rand()?"
        )
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["script_content"])
                is_correct, msg = validator.check_x()
                self.assertFalse(is_correct)
                self.assertEqual(msg, error_msg)


if __name__ == "__main__":
    unittest.main()

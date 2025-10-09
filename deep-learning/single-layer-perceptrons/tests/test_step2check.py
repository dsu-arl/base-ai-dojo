"""Unit tests for the step_2_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep2Check(BaseTestValidator):
    """Unit tests for the step_2_check method of the Validator class.

    This class tests the validation logic of the step_2_check method, which verifies
    that the user's code correctly defines two np.arrays with the stated sample data.
    Inherits from BaseTestValidator to reuse common setup and utility methods for
    creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_2_check with valid code that correctly defines np.arrays 'X' and
        'y' with the stated sample data.

        Verifies that step_2_check returns (True, "") when the code defines np.arrays
        'X' and 'y' that contain sample data that simulate an OR gate and that the
        variables 'X' and 'y' are stored in validator.user_vars.x and
        validator.user_vars.y.
        """
        content = """
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")
        self.assertEqual(validator.user_vars.x, "X")
        self.assertEqual(validator.user_vars.y, "y")

    def test_missing_nparray(self):
        """Tests step_2_check when np.array is not called in the code.

        Verifies that step_2_check returns (False, "np.array isn't called data") when
        the code doesn't contain an np.array call containing the data.
        """
        content = ""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "np.array isn't called")

    def test_nparray_called_more_than_twice(self):
        """Tests step_2_check when np.array is called more than twice.

        Verifies that step_2_check returns (False, "np.array should only be called
        twice, once for X and once for y") when the code contains more than two calls to
        np.array.
        """
        content = """
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "np.array should only be called twice, once for X and once for y"
        )

    def test_assignment_to_variable(self):
        """Tests step_2_check when np.array calls are not assigned to variables.

        Verifies that step_2_check returns (False, "np.array should be assigned to X and
        y variables.") when the code is missing assignments for either of the input or
        output np.array calls. Uses subTests to check multiple error cases.
        """
        cases = [
            {
                "desc": "X not assigned, y assigned",
                "content": """
np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])
""",
            },
            {
                "desc": "X assigned, y not assigned",
                "content": """
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
np.array([[0], [1], [1], [1]])
""",
            },
            {
                "desc": "Neither assigned",
                "content": """
np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
np.array([[0], [1], [1], [1]])
""",
            },
        ]

        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_2_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg, "np.array should be assigned to X and y variables."
                )

    def test_incorrect_x_nparray_args(self):
        """Tests step_2_check when input data passed to np.array is incorrect.

        Verifies that step_2_check returns (False, "Input data passed to np.array
        doesn't match instructions.") when the data passed to np.array for the input is
        doesn't match the instructions.
        """
        content = """
X = np.array([])
y = np.array([[0], [1], [1], [1]])
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Input data passed to np.array doesn't match instructions."
        )

    def test_incorrect_y_nparray_args(self):
        """Tests step_2_check when output data passed to np.array is incorrect.

        Verifies that step_2_check returns (False, "Output data passed to np.array
        doesn't match instructions.") when the data passed to np.array for the output is
        doesn't match the instructions.
        """
        content = """
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([])
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Output data passed to np.array doesn't match instructions."
        )


if __name__ == "__main__":
    unittest.main()

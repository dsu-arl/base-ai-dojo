"""Unit tests for the step_3_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep3Check(BaseTestValidator):
    """Unit tests for the step_3_check method of the Validator class.

    This class tests the validation logic of the step_3_check method, which verifies
    that the user's code correctly two variables to store the OR operation sample data.
    Inherits from BaseTestValidator to reuse common setup and utility methods for
    creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_3_check with valid code that correctly defines two variables to
        store the OR operation sample data in np.arrays.

        Verifies that step_3_check returns (True, "") when the code defines two
        np.arrays, one for the input data [[0, 0], [0, 1], [1, 0], [1, 1]] and one for
        the output data [[0], [0], [0], [1]], both having a datatype of np.float32, and
        storing the variable names in validator.data_user_vars.x and
        validator.data_user_vars.y.
        """
        content = """
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_3_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")
        self.assertEqual(validator.data_user_vars.x, "X")
        self.assertEqual(validator.data_user_vars.y, "y")

    def test_incorrect_x_nparray(self):
        """Tests step_3_check when np.array for input data is not called or incorrect.

        Verifies that step_3_check returns (False, "Missing or incorrect np.array
        statement for input data") when the code doesn't contain an np.array call
        containing the input data or the data or dtype parameter are incorrect. Uses
        subTests to check multiple error cases.
        """
        cases = [
            {
                "desc": "Missing np.array call",
                "content": """
y = np.array([[0], [0], [0], [1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)
""",
            },
            {
                "desc": "Incorrect data",
                "content": """
X = np.array([], dtype=np.int32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)
""",
            },
            {
                "desc": "Incorrect dtype parameter",
                "content": """
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_3_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg, "Missing or incorrect np.array statement for input data"
                )

    def test_incorrect_y_nparray(self):
        """Tests step_3_check when np.array for output data is not called or incorrect.

        Verifies that step_3_check returns (False, "Missing or incorrect np.array
        statement for output data") when the code doesn't contain an np.array call
        containing the output data or the data or dtype parameter are incorrect. Uses
        subTests to check multiple error cases.
        """
        cases = [
            {
                "desc": "Missing np.array call",
                "content": """
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
""",
            },
            {
                "desc": "Incorrect data",
                "content": """
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([], dtype=np.float32)
""",
            },
            {
                "desc": "Incorrect dtype parameter",
                "content": """
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.int32)
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_3_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg, "Missing or incorrect np.array statement for output data"
                )

    def test_nparray_called_more_than_twice(self):
        """Tests step_3_check when np.array is called more than twice.

        Verifies that step_3_check returns (False, "np.array should only be called
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
        is_correct, msg = validator.step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "np.array should only be called twice, once for X and once for y"
        )


if __name__ == "__main__":
    unittest.main()

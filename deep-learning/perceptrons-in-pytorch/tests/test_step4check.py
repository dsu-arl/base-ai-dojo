"""Unit tests for the step_4_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep4Check(BaseTestValidator):
    """Unit tests for the step_4_check method of the Validator class.

    This class tests the validation logic of the step_4_check method, which verifies
    that the user's code correctly converts the input and output data from NumPy arrays
    into PyTorch tensors. Inherits from BaseTestValidator to reuse common setup and
    utility methods for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_4_check with valid code that converts the input and output data to
        PyTorch tensors.

        Verifies that step_4_check returns (True, "") when the code calls
        torch.from_numpy to convert the input and output data from NumPy arrays into
        PyTorch tensors, and storing the variable names in validator.data_user_vars.x
        and validator.data_user_vars.y in case the user renames their variables.
        """
        content = """
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)
"""
        validator = self.create_validator(content)
        validator.data_user_vars.x = "X"
        validator.data_user_vars.y = "y"
        is_correct, msg = validator.step_4_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")
        self.assertEqual(validator.data_user_vars.x, "X_tensor")
        self.assertEqual(validator.data_user_vars.y, "y_tensor")

    def test_missing_x_tensor(self):
        """Tests step_4_check when the input data is missing the torch.from_numpy call.

        Verifies that step_4_check returns (False, "Missing input data conversion to
        PyTorch tensor") when the code doesn't contain a torch.from_numpy call for the
        input data.
        """
        content = "y = torch.from_numpy(y)"
        validator = self.create_validator(content)
        validator.data_user_vars.x = "X"
        validator.data_user_vars.y = "y"
        is_correct, msg = validator.step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Missing input data conversion to PyTorch tensor")

    def test_missing_y_tensor(self):
        """Tests step_4_check when the output data is missing the torch.from_numpy call.

        Verifies that step_4_check returns (False, "Missing output data conversion to
        PyTorch tensor") when the code doesn't contain a torch.from_numpy call for the
        output data.
        """
        content = "X = torch.from_numpy(X)"
        validator = self.create_validator(content)
        validator.data_user_vars.x = "X"
        validator.data_user_vars.y = "y"
        is_correct, msg = validator.step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Missing output data conversion to PyTorch tensor")


if __name__ == "__main__":
    unittest.main()

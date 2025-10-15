"""Unit tests for the step_2_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep2Check(BaseTestValidator):
    """Unit tests for the step_2_check method of the Validator class.

    This class tests the validation logic of the step_2_check method, which verifies
    that the user's code correctly defines a Perceptron class containing __init__ and
    forward methods. Inherits from BaseTestValidator to reuse common setup and utility
    methods for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_2_check with valid code that correctly defines a Perceptron class
        and its two methods, __init__ and forward.

        Verifies that step_2_check returns (True, "") when the code defines a class
        called Perceptron that inherits from nn.Module, defines a __init__ method that
        calls super() and has a linear and sigmoid attribute, and defines a forward
        method that takes in a value and passes it through the linear and sigmoid
        attributes, and stores linear and sigmoid attributes and x parameter in
        validator.class_user_vars.
        """
        content = """
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")
        self.assertEqual(validator.class_user_vars.linear, "self.linear")
        self.assertEqual(validator.class_user_vars.sigmoid, "self.sigmoid")
        self.assertEqual(validator.class_user_vars.x, "x")

    def test_missing_perceptron_class(self):
        """Tests step_2_check when the code doesn't contain a class definition called
        Perceptron.

        Verifies that step_2_check returns (False, "Missing or incorrectly defined
        Perceptron class") when the code doesn't contain a class called Perceptron.
        """
        cases = [
            {
                "desc": "Missing Perceptron class",
                "content": "",
            },
            {
                "desc": "Missing nn.Module inheritance",
                "content": """
class Perceptron():
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_2_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, "Missing or incorrectly defined Perceptron class")

    def test_missing_init_method(self):
        """Tests step_2_check with code that is missing __init__ method of the
        Perceptron class.

        Verifies that step_2_check returns (False, "Missing __init__ method in
        Perceptron class") when the __init__ method isn't defined in the Perceptron
        class.
        """
        content = """
class Perceptron(nn.Module):
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Missing __init__ method in Perceptron class")

    def test_incorrect_init_method(self):
        """Tests step_2_check with code that incorrectly defines the __init__ method of
        the Perceptron class.

        Verifies that step_2_check returns (False, "Incorrectly defined __init__ method
        in Perceptron class") when the __init__ method doesn't contain the correct
        linear and sigmoid attributes. Uses subTests to check multiple error cases.
        """
        cases = [
            {
                "desc": "Has parameters in __init__ besides self",
                "content": """
class Perceptron(nn.Module):
    def __init__(self, test):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
""",
            },
            {
                "desc": "Missing super() call",
                "content": """
class Perceptron(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
""",
            },
            {
                "desc": "Missing nn.Linear call",
                "content": """
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
""",
            },
            {
                "desc": "Missing nn.Sigmoid call",
                "content": """
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_2_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "Incorrectly defined __init__ method in Perceptron class",
                )

    def test_missing_forward_method(self):
        """Tests step_2_check with code that is missing forward method of the Perceptron
        class.

        Verifies that step_2_check returns (False, "Missing forward method in Perceptron
        class") when the forward method isn't defined in the Perceptron class.
        """
        content = """
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Missing forward method in Perceptron class")

    def test_incorrect_forward_method(self):
        """Tests step_2_check with code that incorrectly defines the forward method of
        the Perceptron class.

        Verifies that step_2_check returns (False, "Incorrectly defined
        forward method in Perceptron class") when the forward method contain calls to
        the linear and sigmoid attributes, or doesn't return x. Uses subTests to check
        multiple error cases.
        """
        cases = [
            {
                "desc": "Has more than self and x for parameters",
                "content": """
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, test):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
""",
            },
            {
                "desc": "Missing self.linear call",
                "content": """
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(x)
        return x
""",
            },
            {
                "desc": "Missing self.sigmoid call",
                "content": """
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return x
""",
            },
            {
                "desc": "Missing return",
                "content": """
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_2_check()
                validator.class_user_vars.linear = "self.linear"
                validator.class_user_vars.sigmoid = "self.sigmoid"
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "Incorrectly defined forward method in Perceptron class",
                )


if __name__ == "__main__":
    unittest.main()

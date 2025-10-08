"""Unit tests for the step_3_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep3Check(BaseTestValidator):
    """Unit tests for the step_3_check method of the Validator class.

    This class tests the validation logic of the step_3_check method, which verifies
    that the user's code correctly defines the perceptron defined in the instructions.
    Inherits from BaseTestValidator to reuse common setup and utility methods for
    creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_3_check with valid code that correctly defines a perceptron.

        Verifies that step_3_check returns (True, "") when the code creates a perceptron
        using TensorFlow with an input layer accepting 2 inputs and an output layer
        outputting a single value. Also verifies that the variable 'model' is correctly
        stored in validator.user_vars.model.
        """
        content = """
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_3_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")
        self.assertEqual(validator.user_vars.model, "model")

    def test_function_not_called(self):
        """Tests step_3_check when tf.keras.Sequential is not called in the code.

        Verifies that step_3_check returns (False, "tf.keras.Sequential() isn't called")
        when the code doesn't contain a call to tf.keras.Sequential.
        """
        content = ""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "tf.keras.Sequential() isn't called")

    def test_function_more_than_once(self):
        """Tests step_3_check when tf.keras.Sequential is called more than once.

        Verifies that step_3_check returns (True, "tf.keras.Sequential() should only be
        called once") when the code contains more than one call to tf.keras.Sequential.
        """
        content = """
model = tf.keras.Sequential()
model2 = tf.keras.Sequential()
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "tf.keras.Sequential() should only be called once")

    def test_output_not_assigned(self):
        """Tests step_3_check when the output of tf.keras.Sequential is not assigned to
        a variable.

        Verifies that step_3_check returns (False, "Make sure you store the output of
        tf.keras.Sequential() in a variable") when the output of tf.keras.Sequential is
        not stored in a variable.
        """
        content = """
tf.keras.Sequential()
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Make sure you store the output of tf.keras.Sequential() in a variable"
        )

    def test_incorrect_model_args(self):
        """Tests step_3_check when the layer of the perceptron are incorrect.

        Verifies that step_3_check returns (False, "Missing or incorrect layers for
        perceptron") when the layers of the perceptron are missing or incorrect. Uses
        subTests to check multiple error cases.
        """
        cases = [
            {
                "desc": "missing Input layer",
                "content": """
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
""",
            },
            {
                "desc": "missing Dense layer parameters",
                "content": """
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense()
])
""",
            },
            {
                "desc": "incorrect units parameter value",
                "content": """
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(units=5, activation='sigmoid')
])
""",
            },
            {
                "desc": "incorrect input shape parameter value",
                "content": """
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
""",
            },
            {
                "desc": "incorrect activation parameter value",
                "content": """
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(units=1, activation='relu')
])
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_3_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, "Missing or incorrect layers for perceptron")


if __name__ == "__main__":
    unittest.main()

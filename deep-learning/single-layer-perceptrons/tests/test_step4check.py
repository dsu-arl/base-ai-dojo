"""Unit tests for the step_4_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep4Check(BaseTestValidator):
    """Unit tests for the step_4_check method of the Validator class.

    This class tests the validation logic of the step_4_check method, which verifies
    that the user's code correctly states to use the 'adam' optimizer and
    'binary_crossentropy' for the loss for the model. Inherits from BaseTestValidator to
    reuse common setup and utility methods for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_4_check with valid code that correctly compiles the model.

        Verifies that step_4_check returns (True, "") when the .compile function is
        called on the model with the 'adam' optimizer and 'binary_crossentropy' for the
        loss function.
        """
        content = """
model.compile(optimizer='adam', loss='binary_crossentropy')
"""
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_4_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")

    def test_compile_not_called(self):
        """Tests step_4_check when model.compile is not called in the code.

        Verifies that step_4_check returns (False, "model.compile() isn't called")
        when the code doesn't contain a call to model.compile.
        """
        content = ""
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.compile() isn't called")

    def test_compile_called_more_than_once(self):
        """Tests step_4_check when model.compile is called more than once.

        Verifies that step_4_check returns (True, "model.compile() should only be called
        once") when the code contains more than one call to model.compile.
        """
        content = """
model.compile(optimizer='adam', loss='binary_crossentropy')
model.compile(optimizer='adam', loss='binary_crossentropy')
"""
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.compile() should only be called once")

    def test_compile_output_assigned_to_variable(self):
        """Tests step_4_check when the output of model.compile is assigned to a variable

        Verifies that step_4_check returns (False, "model.compile() shouldn't be
        assigned to any variables") when the output of tf.keras.Sequential is stored in
        a variable.
        """
        content = """
test = model.compile(optimizer='adam', loss='binary_crossentropy')
"""
        validator = self.create_validator(content)
        validator.user_vars.model = "model"
        is_correct, msg = validator.step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.compile() shouldn't be assigned to any variables")

    def test_incorrect_compile_params(self):
        """Tests step_4_check when either the optimizer or loss parameters are incorrect

        Verifies that step_4_check returns (False, "Missing or incorrect optimizer and
        loss function passed to model.compile()") when either the optimizer or loss
        parameters are incorrect. Uses subTests to check multiple error cases.
        """
        cases = [
            {
                "desc": "incorrect optimizer",
                "content": """
model.compile(optimizer='', loss='binary_crossentropy')
""",
            },
            {
                "desc": "incorrect loss function",
                "content": """
model.compile(optimizer='adam', loss='')
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.user_vars.model = "model"
                is_correct, msg = validator.step_4_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "Missing or incorrect optimizer and loss "
                    "function passed to model.compile()",
                )


if __name__ == "__main__":
    unittest.main()

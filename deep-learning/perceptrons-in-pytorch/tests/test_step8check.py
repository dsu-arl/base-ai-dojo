"""Unit tests for the step_8_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep8Check(BaseTestValidator):
    """Unit tests for the step_8_check method of the Validator class.

    This class tests the validation logic of the step_8_check method, which verifies
    that the user's code correctly makes predictions using the trained model. Inherits
    from BaseTestValidator to reuse common setup and utility methods for creating
    Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_8_check with valid code that correctly makes predictions using the
        trained model.

        Verifies that step_8_check returns (True, "") when the code correctly makes
        predictions using the trained model, rounds the predictions, and stores the
        predictions in validator.data_user_vars.predictions.
        """
        content = """
with torch.no_grad():
    predictions = model(X)
    predictions = predictions.round()
"""
        validator = self.create_validator(content)
        validator.data_user_vars.x = "X"
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_8_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")

    def test_missing_torch_nograd(self):
        """Tests step_8_check when the code doesn't disable gradient computation.

        Verifies that step_8_check returns (False, "Missing code to disable gradient
        computation for inference") when the code doesn't call torch.no_grad to disable
        gradient computation when making predictions using the trained model.
        """
        content = ""
        validator = self.create_validator(content)
        validator.data_user_vars.x = "X"
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Missing code to disable gradient computation for inference"
        )

    def test_missing_inference(self):
        """Tests step_8_check when the code doesn't perform inference.

        Verifies that step_8_check returns (False, "Missing code to make predictions
        using the trained model") when the code doesn't call model(X).
        """
        content = """
with torch.no_grad():
    print()
"""
        validator = self.create_validator(content)
        validator.data_user_vars.x = "X"
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Missing code to make predictions using the trained model"
        )

    def test_inference_output_not_stored(self):
        """Tests step_8_check when predictions are not stored in a variable.

        Verifies that step_8_check returns (False, "Prediction output not stored in a
        variable") when the output of model(X) is not stored in a variable.
        """
        content = """
with torch.no_grad():
    model(X)
"""
        validator = self.create_validator(content)
        validator.data_user_vars.x = "X"
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Prediction output not stored in a variable")

    def test_missing_round(self):
        """Tests step_8_check when predictions are not rounded.

        Verifies that step_8_check returns (False, "Missing predictions.round() call to
        round predictions") when the code doesn't round the outputted predictions.
        """
        content = """
with torch.no_grad():
    predictions = model(X)
"""
        validator = self.create_validator(content)
        validator.data_user_vars.x = "X"
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Missing predictions.round() call to round predictions")

    def test_round_output_not_stored(self):
        """Tests step_8_check when rounded predictions are not stored in a variable.

        Verifies that step_8_check returns (False, "predictions.round() output not
        stored in a variable") when the output of predictions.round is not stored in a
        variable, either a new one or the existing one.
        """
        content = """
with torch.no_grad():
    predictions = model(X)
    predictions.round()
"""
        validator = self.create_validator(content)
        validator.data_user_vars.x = "X"
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "predictions.round() output not stored in a variable")

    def test_incorrect_inference_data(self):
        """Tests step_8_check when the data passed to the model for inference is
        incorrect.

        Verifies that step_8_check returns (False, "Incorrect data passed to model for
        inference") when the data passed to model is not X.
        """
        content = """
with torch.no_grad():
    predictions = model(y)
    predictions = predictions.round()
"""
        validator = self.create_validator(content)
        validator.data_user_vars.x = "X"
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Incorrect data passed to model for inference")

    def test_round_has_args(self):
        """Tests step_8_check when predictions.round is given arguments.

        Verifies that step_8_check returns (False, "You shouldn't be passing any
        parameters to predictions.round() for this challenge") when predictions.round is
        called with arguments. Uses subTests to check multiple argument error cases.
        """
        cases = [
            {
                "desc": "Has args",
                "content": """
with torch.no_grad():
    predictions = model(X)
    predictions = predictions.round(23)
""",
            },
            {
                "desc": "Has kwargs",
                "content": """
with torch.no_grad():
    predictions = model(X)
    predictions = predictions.round(test=23)
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.data_user_vars.x = "X"
                validator.model_user_vars.model = "model"
                is_correct, msg = validator.step_8_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "You shouldn't be passing any parameters to predictions.round() "
                    "for this challenge",
                )


if __name__ == "__main__":
    unittest.main()

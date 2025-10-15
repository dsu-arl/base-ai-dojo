"""Unit tests for the step_6_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep6Check(BaseTestValidator):
    """Unit tests for the step_6_check method of the Validator class.

    This class tests the validation logic of the step_6_check method, which verifies
    that the user's code correctly defines the loss function and optimizer. Inherits
    from BaseTestValidator to reuse common setup and utility methods for creating
    Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_6_check with valid code that defines the loss function and
        optimizer.

        Verifies that step_6_check returns (True, "") when the code creates a variable
        for binary cross-entropy loss and a variable for the Adam optimizer, and stores
        the variable names in validator.model_user_vars.
        """
        content = """
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
"""
        validator = self.create_validator(content)
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_6_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")
        self.assertEqual(validator.model_user_vars.criterion, "criterion")
        self.assertEqual(validator.model_user_vars.optimizer, "optimizer")

    def test_loss_function_not_called(self):
        """Tests step_6_check when nn.BCELoss is not called.

        Verifies that step_6_check returns (False, "nn.BCELoss() isn't called") when the
        code doesn't call nn.BCELoss.
        """
        content = "optimizer = torch.optim.Adam(model.parameters())"
        validator = self.create_validator(content)
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "nn.BCELoss() isn't called")

    def test_optimizer_not_called(self):
        """Tests step_6_check when torch.optim.Adam is not called.

        Verifies that step_6_check returns (False, "torch.optim.Adam() isn't called")
        when the code doesn't call torch.optim.Adam.
        """
        content = "criterion = nn.BCELoss()"
        validator = self.create_validator(content)
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "torch.optim.Adam() isn't called")

    def test_loss_function_called_more_than_once(self):
        """Tests step_6_check when nn.BCELoss is called more than once.

        Verifies that step_6_check returns (False, "nn.BCELoss() should only be called
        once") when the code contains multiple calls to nn.BCELoss.
        """
        content = """
criterion = nn.BCELoss()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_6_check()
        validator.model_user_vars.model = "model"
        self.assertFalse(is_correct)
        self.assertEqual(msg, "nn.BCELoss() should only be called once")

    def test_optimizer_called_more_than_once(self):
        """Tests step_6_check when torch.optim.Adam is called more than once.

        Verifies that step_6_check returns (False, "torch.optim.Adam() should only be
        called once") when the code contains multiple calls to torch.optim.Adam.
        """
        content = """
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.Adam(model.parameters())
"""
        validator = self.create_validator(content)
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "torch.optim.Adam() should only be called once")

    def test_loss_function_output_not_assigned(self):
        """Tests step_6_check when output of nn.BCELoss is not stored in any variables.

        Verifies that step_6_check returns (False, "Make sure you store the output of
        nn.BCELoss() in a variable") when the code calls nn.BCELoss with no output
        variable specified.
        """
        content = """
nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
"""
        validator = self.create_validator(content)
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Make sure you store the output of nn.BCELoss() in a variable"
        )

    def test_optimizer_output_not_assigned(self):
        """Tests step_6_check when output of torch.optim.Adam is not stored in any
        variables.

        Verifies that step_6_check returns (False, "Make sure you store the output of
        torch.optim.Adam() in a variable") when the code calls torch.optim.Adam with no
        output variable specified.
        """
        content = """
criterion = nn.BCELoss()
torch.optim.Adam(model.parameters())
"""
        validator = self.create_validator(content)
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg, "Make sure you store the output of torch.optim.Adam() in a variable"
        )

    def test_loss_function_has_args(self):
        """Tests step_6_check when nn.BCELoss is called with arguments.

        Verifies that step_6_check returns (False, "You shouldn't be passing any
        parameters to nn.BCELoss() for this challenge") when nn.BCELoss is called with
        arguments. Uses subTests to check multiple argument error cases.
        """
        cases = [
            {
                "desc": "Has args",
                "content": """
criterion = nn.BCELoss(23)
optimizer = torch.optim.Adam(model.parameters())
""",
            },
            {
                "desc": "Has kwargs",
                "content": """
criterion = nn.BCELoss(test=23)
optimizer = torch.optim.Adam(model.parameters())
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.model_user_vars.model = "model"
                is_correct, msg = validator.step_6_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg,
                    "You shouldn't be passing any parameters to nn.BCELoss() "
                    "for this challenge",
                )

    def test_optimizer_incorrect_args(self):
        """Tests step_6_check when torch.optim.Adam is called with incorrect arguments.

        Verifies that step_6_check returns (False, "Incorrect parameters for
        torch.optim.Adam(), are you correctly passing the model parameters to it?") when
        torch.optim.Adam is called with incorrect arguments.
        """
        content = """
criterion = nn.BCELoss()
optimizer = torch.optim.Adam()
"""
        validator = self.create_validator(content)
        validator.model_user_vars.model = "model"
        is_correct, msg = validator.step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(
            msg,
            "Incorrect parameters for torch.optim.Adam(), are you correctly passing "
            "the model parameters to it?",
        )


if __name__ == "__main__":
    unittest.main()

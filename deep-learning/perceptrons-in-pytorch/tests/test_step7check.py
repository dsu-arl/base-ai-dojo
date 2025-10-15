"""Unit tests for the step_7_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep7Check(BaseTestValidator):
    """Unit tests for the step_7_check method of the Validator class.

    This class tests the validation logic of the step_7_check method, which verifies
    that the user's code correctly creates the training loop to train the model.
    Inherits from BaseTestValidator to reuse common setup and utility methods for
    creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_7_check with valid code that has a correct training loop.

        Verifies that step_7_check returns (True, "") when the code defines a training
        loop that performs a forward pass, backward pass, and optimization to train the
        model. Uses subTests to check multiple correct cases.
        """
        cases = [
            {
                "desc": "Epochs variable",
                "content": """
epochs = 100
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
""",
            },
            {
                "desc": "No epochs variable",
                "content": """
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.data_user_vars.x = "X"
                validator.data_user_vars.y = "y"
                validator.model_user_vars.model = "model"
                validator.model_user_vars.criterion = "criterion"
                validator.model_user_vars.optimizer = "optimizer"
                is_correct, msg = validator.step_7_check()
                self.assertTrue(is_correct)
                self.assertEqual(msg, "")

    def test_missing_training_loop(self):
        """Tests step_7_check when the training loop is missing.

        Verifies that step_7_check returns (False, "Missing training loop") when the
        code is missing the for loop for training. Uses subTests to check mutliple for
        loop errors.
        """
        cases = [
            {"desc": "No for loop", "content": ""},
            {
                "desc": "Not using range()",
                "content": """
for epoch in epochs:
    print()
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_7_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, "Missing training loop")

    def test_incorrect_training_loop_order(self):
        """Tests step_7_check when the contents of the training loop are out of order.

        Verifies that step_7_check returns (False, "Incorrect training loop") when the
        contents of the training loop are out of order.
        """
        content = """
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
"""
        validator = self.create_validator(content)
        validator.data_user_vars.x = "X"
        validator.data_user_vars.y = "y"
        validator.model_user_vars.model = "model"
        validator.model_user_vars.criterion = "criterion"
        validator.model_user_vars.optimizer = "optimizer"
        is_correct, msg = validator.step_7_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Incorrect training loop")

    def test_incorrect_epoch_count(self):
        """Tests step_7_check when the training loop doesn't run for 100 epochs.

        Verifies that step_7_check returns (False, "Incorrect epoch count for training
        loop") when the training loop isn't set to run for 100 epochs. Uses subTests for
        when epochs are defined as variables or in for loop declaration.
        """
        cases = [
            {
                "desc": "Incorrect epochs variable",
                "content": """
epochs = 5
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
""",
            },
            {
                "desc": "Incorrect epochs in for loop declaration",
                "content": """
for epoch in range(5):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.data_user_vars.x = "X"
                validator.data_user_vars.y = "y"
                validator.model_user_vars.model = "model"
                validator.model_user_vars.criterion = "criterion"
                validator.model_user_vars.optimizer = "optimizer"
                is_correct, msg = validator.step_7_check()
                self.assertEqual(msg, "Incorrect epoch count for training loop")
                self.assertFalse(is_correct)

    def test_incorrect_forward_pass(self):
        """Tests step_7_check when the training loop is missing the forward pass.

        Verifies that step_7_check returns (False, "Missing or incorrect forward pass in
        training loop") when the training loop doesn't contain a forward pass or the
        forward pass is incorrect. Uses subTests to check multiple forward pass errors.
        """
        cases = [
            {
                "desc": "Missing forward pass",
                "content": """
for epoch in range(100):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
""",
            },
            {
                "desc": "Missing model predictions",
                "content": """
for epoch in range(100):
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
""",
            },
            {
                "desc": "Missing loss calculation",
                "content": """
for epoch in range(100):
    outputs = model(X)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.data_user_vars.x = "X"
                validator.data_user_vars.y = "y"
                validator.model_user_vars.model = "model"
                validator.model_user_vars.criterion = "criterion"
                validator.model_user_vars.optimizer = "optimizer"
                is_correct, msg = validator.step_7_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg, "Missing or incorrect forward pass in training loop"
                )

    def test_incorrect_backward_pass(self):
        """Tests step_7_check when the training loop is missing the backward pass.

        Verifies that step_7_check returns (False, "Missing or incorrect backward pass
        in training loop") when the training loop doesn't contain a backward pass or the
        backward pass is incorrect. Uses subTests to check multiple backward pass errors
        """
        cases = [
            {
                "desc": "Missing backward pass",
                "content": """
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)
""",
            },
            {
                "desc": "Missing clear previous gradients",
                "content": """
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()
""",
            },
            {
                "desc": "Missing compute gradients",
                "content": """
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    optimizer.step()
""",
            },
            {
                "desc": "Missing update weights",
                "content": """
for epoch in range(100):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                validator.data_user_vars.x = "X"
                validator.data_user_vars.y = "y"
                validator.model_user_vars.model = "model"
                validator.model_user_vars.criterion = "criterion"
                validator.model_user_vars.optimizer = "optimizer"
                is_correct, msg = validator.step_7_check()
                self.assertFalse(is_correct)
                self.assertEqual(
                    msg, "Missing or incorrect backward pass in training loop"
                )


if __name__ == "__main__":
    unittest.main()

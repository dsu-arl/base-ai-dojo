"""Unit tests for the step_1_check function."""

import unittest
from base_test import BaseTestValidator


class TestStep1Check(BaseTestValidator):
    """Unit tests for the step_1_check method of the Validator class.

    This class tests the validation logic of the step_1_check method, which verifies
    that the user's code correctly imports the tensorflow library with the 'tf' alias
    and the numpy library with  the 'np' alias. Inherits from BaseTestValidator to reuse
    common setup and utility methods for creating Validator instances.

    Args:
        BaseTestValidator (class): Base class providing shared test utilities, including
            the create_validator method.
    """

    def test_success(self):
        """Tests step_1_check with valid code that correct imports the tensorflow and
        numpy libraries with the correct aliases.

        Verifies that step_1_check returns (True, "") when the script imports tensorflow
        with the 'tf' alias and numpy with the 'np' alias.
        """
        content = """
import tensorflow as tf
import numpy as np
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_1_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, "")

    def test_missing_tensorflow_import(self):
        """Tests step_1_check when code is missing the tensorflow import.

        Verifies that step_1_check returns (False, "Missing or incorrect tensorflow
        import") when the code does not import the tensorflow library.
        """
        content = """
# Missing tensorflow
import numpy as np
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Missing or incorrect tensorflow import")

    def test_missing_numpy_import(self):
        """Tests step_1_check when code is missing the numpy import.

        Verifies that step_1_check returns (False, "Missing or incorrect numpy import")
        when the code does not import the numpy library.
        """
        content = """
import tensorflow as tf
# Missing numpy
"""
        validator = self.create_validator(content)
        is_correct, msg = validator.step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Missing or incorrect numpy import")

    def test_incorrect_tensorflow_alias(self):
        """Tests step_1_check when code imports tensorflow, but without an alias or not
        the specified alias.

        Verifies that step_1_check returns (False, "Missing or incorrect tensorflow
        import alias") when the code imports the tensorflow library, but without an
        alias or not the specified alias.
        """
        cases = [
            {
                "desc": "Missing import alias",
                "content": """
import tensorflow
import numpy as np
""",
            },
            {
                "desc": "Incorrect import alias",
                "content": """
import tensorflow as test
import numpy as np
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_1_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, "Missing or incorrect tensorflow import alias")

    def test_incorrect_numpy_alias(self):
        """Tests step_1_check when code imports numpy, but without an alias or not the
        specified alias.

        Verifies that step_1_check returns (False, "Missing or incorrect numpy import
        alias") when the code imports the tensorflow library, but without an alias or
        not the specified alias.
        """
        cases = [
            {
                "desc": "Missing import alias",
                "content": """
import tensorflow as tf
import numpy
""",
            },
            {
                "desc": "Incorrect import alias",
                "content": """
import tensorflow as tf
import numpy as test
""",
            },
        ]
        for case in cases:
            with self.subTest(case=case["desc"]):
                validator = self.create_validator(case["content"])
                is_correct, msg = validator.step_1_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, "Missing or incorrect numpy import alias")


if __name__ == "__main__":
    unittest.main()

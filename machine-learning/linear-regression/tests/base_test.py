"""Base test validator class that contains functionality that is used across all unit
tests.
"""

import os
from tempfile import NamedTemporaryFile
import unittest
from verify import Validator


class BaseTestValidator(unittest.TestCase):
    """Base class for unit tests to inherit from that includes reused functionality.

    Args:
        unittest.TestCase (class): A class whose instances are single test cases.
    """

    def create_validator(self, script_content: str) -> Validator:
        """Creates the validator instance for other unit tests to use.

        Args:
            script_content (str): Python code to initialize Validator object with.

        Raises:
            Exception: Removes temporary file if any error occurs.

        Returns:
            Validator: Validator object to use in unit tests.
        """
        tmp_filename = None
        try:
            with NamedTemporaryFile(
                suffix=".py", mode="w+", encoding="utf-8", delete=False
            ) as tmp_file:
                tmp_file.write(script_content)
                tmp_file.flush()
                tmp_filename = tmp_file.name
            validator = Validator(tmp_filename)
        except Exception:
            if tmp_filename and os.path.exists(tmp_filename):
                os.remove(tmp_filename)
            raise
        self.addClassCleanup(lambda: os.remove(tmp_filename))

        return validator

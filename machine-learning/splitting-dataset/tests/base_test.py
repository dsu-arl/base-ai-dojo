import unittest
from tempfile import NamedTemporaryFile
from verify import Validator
import os


class BaseTestValidator(unittest.TestCase):
    def create_validator(self, script_content: str) -> Validator:
        tmp_file = NamedTemporaryFile(suffix='.py', mode='w+', encoding='utf-8', delete=False)
        try:
            tmp_file.write(script_content)
            tmp_file.flush()
            tmp_file.close()
            validator = Validator(tmp_file.name)
        except Exception as e:
            os.remove(tmp_file.name)
            raise e
        self.addCleanup(lambda: os.remove(tmp_file.name))
        return validator

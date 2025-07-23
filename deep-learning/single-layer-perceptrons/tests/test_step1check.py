from base_test import BaseTestValidator
import unittest


class TestStep1Check(BaseTestValidator):
    def test_success(self):
        script_content = """
import tensorflow as tf
import numpy as np
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_1_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
    
    def test_missing_tensorflow_import(self):
        script_content = """
# Missing tensorflow
import numpy as np
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'Missing or incorrect tensorflow import line, did you import it with the specified alias?')

    def test_missing_numpy_import(self):
        script_content = """
import tensorflow as tf
# Missing numpy
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'Missing or incorrect numpy import line, did you import it with the specified alias?')
    
    def test_incorrect_tensorflow_alias(self):
        script_content = """
import tensorflow
import numpy as np
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'Missing or incorrect tensorflow import line, did you import it with the specified alias?')
    
    def test_incorrect_numpy_alias(self):
        script_content = """
import tensorflow as tf
import numpy
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'Missing or incorrect numpy import line, did you import it with the specified alias?')


if __name__ == '__main__':
    unittest.main()

from base_test import BaseTestValidator
import unittest


class TestStep8Check(BaseTestValidator):
    def test_success(self):
        script_content = """
print(predictions)
"""
        validator = self.create_validator(script_content)
        validator.predictions = 'predictions'
        is_correct, msg = validator._step_8_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
    
    def test_print_not_called(self):
        script_content = ''
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() isn't called")

    def test_print_called_more_than_once(self):
        script_content = """
print()
print()
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() should only be called once")

    def test_incorrect_print_value(self):
        script_content = """
print(X)
"""
        validator = self.create_validator(script_content)
        validator.predictions = 'predictions'
        is_correct, msg = validator._step_8_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Incorrect parameters passed to print(), are you passing your rounded predictions to the print() function?")


if __name__ == '__main__':
    unittest.main()

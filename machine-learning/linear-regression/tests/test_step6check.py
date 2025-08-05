from base_test import BaseTestValidator
import unittest


class TestStep6Check(BaseTestValidator):
    def test_success(self):
        content = 'print(mse)'
        validator = self.create_validator(content)
        validator._mse = 'mse'
        is_correct, msg = validator._step_6_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')

    def test_missing_function_call(self):
        content = ''
        validator = self.create_validator(content)
        is_correct, msg = validator._step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() isn't called")
    
    def test_function_called_more_than_once(self):
        content = """
print(mse)
print(mse)
"""
        validator = self.create_validator(content)
        is_correct, msg = validator._step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() shouldn't be called more than once")

    def test_output_assigned(self):
        content = 'test = print(mse)'
        validator = self.create_validator(content)
        is_correct, msg = validator._step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "print() shouldn't be assigned to any variables")

    def test_incorrect_args(self):
        content = 'print()'
        validator = self.create_validator(content)
        is_correct, msg = validator._step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'Are you printing out the correct variable for the MSE?')
    
    def test_incorrect_kwargs(self):
        content = 'print(mse, test=mse)'
        validator = self.create_validator(content)
        validator._mse = 'mse'
        is_correct, msg = validator._step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "You don't need any keyword arguments for this print statement")


if __name__ == '__main__':
    unittest.main()
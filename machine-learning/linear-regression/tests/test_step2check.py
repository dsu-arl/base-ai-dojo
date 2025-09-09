from base_test import BaseTestValidator
import unittest


class TestStep2Check(BaseTestValidator):
    def test_success(self):
        content = 'model = LinearRegression()'
        validator = self.create_validator(content)
        is_correct, msg = validator._step_2_check()
        self.assertEqual(validator._model, 'model')
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')

    def test_missing_function_call(self):
        content = ''
        validator = self.create_validator(content)
        is_correct, msg = validator._step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "LinearRegression() isn't called")
    
    def test_function_called_more_than_once(self):
        content = """
model = LinearRegression()
model = LinearRegression()
"""
        validator = self.create_validator(content)
        is_correct, msg = validator._step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "LinearRegression() should only be called once")

    def test_output_not_assigned(self):
        content = 'LinearRegression()'
        validator = self.create_validator(content)
        is_correct, msg = validator._step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'LinearRegression() should be assigned to a variable')

    def test_function_has_args(self):
        cases = [
            {
                'desc': 'Has args',
                'content': 'model = LinearRegression(23)'
            },
            {
                'desc': 'Has kwargs',
                'content': 'model = LinearRegression(test=23)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['content'])
                is_correct, msg = validator._step_2_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, "You shouldn't be passing any parameters to LinearRegression() for this challenge")


if __name__ == '__main__':
    unittest.main()
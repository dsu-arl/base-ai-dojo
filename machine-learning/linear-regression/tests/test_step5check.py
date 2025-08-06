from base_test import BaseTestValidator
import unittest


class TestStep5Check(BaseTestValidator):
    def test_success(self):
        cases = [
            {
                'desc': 'Passing parameters using args',
                'content': 'mse = mean_squared_error(y_test, y_pred)'
            },
            {
                'desc': 'Passing parameters using kwargs',
                'content': 'mse = mean_squared_error(y_true=y_test, y_pred=y_pred)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['content'])
                validator._y_test = 'y_test'
                validator._y_pred = 'y_pred'
                is_correct, msg = validator._step_5_check()
                self.assertEqual(validator._mse, 'mse')
                self.assertTrue(is_correct)
                self.assertEqual(msg, '')

    def test_missing_function_call(self):
        content = ''
        validator = self.create_validator(content)
        is_correct, msg = validator._step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "mean_squared_error() isn't called")
    
    def test_function_called_more_than_once(self):
        content = """
mse = mean_squared_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
"""
        validator = self.create_validator(content)
        is_correct, msg = validator._step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "mean_squared_error() shouldn't be called more than once")

    def test_output_not_assigned(self):
        content = 'mean_squared_error(y_test, y_pred)'
        validator = self.create_validator(content)
        is_correct, msg = validator._step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "mean_squared_error() should only be assigned to a single variable")

    def test_incorrect_args(self):
        cases = [
            {
                'desc': 'Empty kwargs with incorrect args',
                'content': 'mse = mean_squared_error(y_train, y_test)'
            },
            {
                'desc': 'Empty args with incorrect kwargs',
                'content': 'mse = mean_squared_error(y_true=y_train, y_pred=y_test)'
            },
            {
                'desc': 'Empty args and empty kwargs',
                'content': 'mse = mean_squared_error()'
            },
            {
                'desc': 'Incorrect number of args',
                'content': 'mse = mean_squared_error(y_true=y_test)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['content'])
                validator._y_test = 'y_test'
                validator._y_pred = 'y_pred'
                is_correct, msg = validator._step_5_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'Incorrect parameters for mean_squared_error(), are you correctly passing the test output and model output to it?')


if __name__ == '__main__':
    unittest.main()
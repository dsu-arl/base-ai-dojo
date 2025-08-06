from base_test import BaseTestValidator
import unittest


class TestStep1Check(BaseTestValidator):
    def test_success(self):
        script_content = 'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)'
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_1_check()
        self.assertEqual(validator._X_train, 'X_train')
        self.assertEqual(validator._X_test, 'X_test')
        self.assertEqual(validator._y_train, 'y_train')
        self.assertEqual(validator._y_test, 'y_test')
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')

    def test_missing_function_call(self):
        script_content = ''
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "train_test_split() isn't called")
    
    def test_function_called_more_than_once(self):
        script_content = """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'train_test_split() should only be called once')

    def test_incorrect_number_output_variables(self):
        cases = [
            {
                'desc': '3 output variables',
                'script_content': 'X_train, X_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)'
            },
            {
                'desc': '5 output variables',
                'script_content': 'X_train, X_test, y_train, y_test, test = train_test_split(X, y, test_size=0.2, random_state=42)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['script_content'])
                is_correct, msg = validator._step_1_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'train_test_split() should be unpacked into 4 variables')

    def test_incorrect_args(self):
        cases = [
            {
                'desc': 'Incorrect X',
                'script_content': 'X_train, X_test, y_train, y_test = train_test_split([], y, test_size=0.2, random_state=42)'
            },
            {
                'desc': 'Incorrect y',
                'script_content': 'X_train, X_test, y_train, y_test = train_test_split(X, test_size=0.2, random_state=42)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['script_content'])
                is_correct, msg = validator._step_1_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'You need to pass X and y to train_test_split() so knows what data to split')

    def test_incorrect_kwargs(self):
        cases = [
            {
                'desc': 'Incorrect test size',
                'script_content': 'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)'
            },
            {
                'desc': 'Incorrect random state',
                'script_content': 'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['script_content'])
                is_correct, msg = validator._step_1_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, "train_test_split() parameters don't match what's expected, did you give 'test_size' and 'random_state' the correct values from the instructions?")


if __name__ == '__main__':
    unittest.main()
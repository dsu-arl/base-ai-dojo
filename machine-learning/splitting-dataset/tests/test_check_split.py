from base_test import BaseTestValidator
import unittest


class TestCheckSplit(BaseTestValidator):
    def test_success(self):
        script_content = 'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)'
        validator = self.create_validator(script_content)
        validator.X = 'X'
        validator.y = 'y'
        is_correct, msg = validator.check_split()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
        self.assertEqual(validator.X_train, 'X_train')
        self.assertEqual(validator.X_test, 'X_test')
        self.assertEqual(validator.y_train, 'y_train')
        self.assertEqual(validator.y_test, 'y_test')

    def test_split_not_called(self):
        script_content = ''
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_split()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "train_test_split() isn't called")
    
    def test_split_called_more_than_once(self):
        script_content = """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
"""
        validator = self.create_validator(script_content)
        validator.X = 'X'
        validator.y = 'y'
        is_correct, msg = validator.check_split()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "train_test_split() should only be called once")
    
    def test_output_not_stored(self):
        script_content = 'train_test_split(X, y, test_size=0.3)'
        validator = self.create_validator(script_content)
        validator.X = 'X'
        validator.y = 'y'
        is_correct, msg = validator.check_split()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Make sure you're storing the output of train_test_split() in variables for your train and test datasets (HINT: It's 4 variables)")

    def test_incorrect_number_output_variables(self):
        cases = [
            {
                'desc': 'Only 3 output variables',
                'script_content': 'X_train, X_test, y_train = train_test_split(X, y, test_size=0.3)'
            },
            {
                'desc': 'Only 2 output variables',
                'script_content': 'X_train, X_test = train_test_split(X, y, test_size=0.3)'
            },
            {
                'desc': 'Only 1 output variable',
                'script_content': 'X_train = train_test_split(X, y, test_size=0.3)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['script_content'])
                validator.X = 'X'
                validator.y = 'y'
                is_correct, msg = validator.check_split()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'Incorrect number of output variables.')

    def test_incorrect_args(self):
        cases = [
            {
                'desc': 'Incorrect X',
                'script_content': 'X_train, X_test, y_train, y_test = train_test_split([], y, test_size=0.3)'
            },
            {
                'desc': 'Incorrect y',
                'script_content': 'X_train, X_test, y_train, y_test = train_test_split(X, test_size=0.3)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['script_content'])
                validator.X = 'X'
                validator.y = 'y'
                is_correct, msg = validator.check_split()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'Did you provide the correct variables for X and y?')

    def test_incorrect_kwargs(self):
        script_content = 'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)'
        validator = self.create_validator(script_content)
        validator.X = 'X'
        validator.y = 'y'
        is_correct, msg = validator.check_split()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Did you correctly set the 'test_size' parameter so that the training dataset is 70% of the original dataset?")


if __name__ == '__main__':
    unittest.main()

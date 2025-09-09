from base_test import BaseTestValidator
import unittest


class TestStep3Check(BaseTestValidator):
    def test_success(self):
        cases = [
            {
                'desc': 'Passing parameters using args',
                'content': 'model.fit(X_train, y_train)'
            },
            {
                'desc': 'Passing parameters using kwargs',
                'content': 'model.fit(X=X_train, y=y_train)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['content'])
                validator._model = 'model'
                validator._X_train = 'X_train'
                validator._y_train = 'y_train'
                is_correct, msg = validator._step_3_check()
                self.assertTrue(is_correct)
                self.assertEqual(msg, '')

    def test_missing_function_call(self):
        content = ''
        validator = self.create_validator(content)
        validator._model = 'model'
        is_correct, msg = validator._step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.fit() isn't called")
    
    def test_function_called_more_than_once(self):
        content = """
model.fit(X_train, y_train)
model.fit(X_train, y_train)
"""
        validator = self.create_validator(content)
        validator._model = 'model'
        is_correct, msg = validator._step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.fit() shouldn't be called more than once")

    def test_output_assigned(self):
        content = 'test = model.fit(X_train, y_train)'
        validator = self.create_validator(content)
        validator._model = 'model'
        is_correct, msg = validator._step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.fit() shouldn't be assigned to any variables")

    def test_incorrect_args(self):
        cases = [
            {
                'desc': 'Empty kwargs with incorrect args',
                'content': 'model.fit(X_test, y_test)'
            },
            {
                'desc': 'Empty args with incorrect kwargs',
                'content': 'model.fit(X=X_test, y=y_test)'
            },
            {
                'desc': 'Non-empty args and non-empty kwargs',
                'content': 'model.fit(X_train, y_train, X=X_train, y=y_train)'
            },
            {
                'desc': 'Empty args and empty kwargs',
                'content': 'model.fit()'
            },
            {
                'desc': 'Incorrect number of args',
                'content': 'model.fit(X_train)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['content'])
                validator._model = 'model'
                is_correct, msg = validator._step_3_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'Incorrect parameters for model.fit(), are you correctly passing the training data to it?')


if __name__ == '__main__':
    unittest.main()
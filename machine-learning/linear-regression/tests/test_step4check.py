from base_test import BaseTestValidator
import unittest


class TestStep4Check(BaseTestValidator):
    def test_success(self):
        cases = [
            {
                'desc': 'Passing parameters using args',
                'content': 'y_pred = model.predict(X_test)'
            },
            {
                'desc': 'Passing parameters using kwargs',
                'content': 'y_pred = model.predict(X=X_test)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['content'])
                validator._model = 'model'
                validator._X_test = 'X_test'
                is_correct, msg = validator._step_4_check()
                self.assertEqual(validator._y_pred, 'y_pred')
                self.assertTrue(is_correct)
                self.assertEqual(msg, '')

    def test_missing_function_call(self):
        content = ''
        validator = self.create_validator(content)
        validator._model = 'model'
        validator._X_test = 'X_test'
        is_correct, msg = validator._step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.predict() isn't called")
    
    def test_function_called_more_than_once(self):
        content = """
y_pred = model.predict(X_test)
y_pred = model.predict(X_test)
"""
        validator = self.create_validator(content)
        validator._model = 'model'
        validator._X_test = 'X_test'
        is_correct, msg = validator._step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.predict() shouldn't be called more than once")

    def test_output_not_assigned(self):
        content = 'model.predict(X_test)'
        validator = self.create_validator(content)
        validator._model = 'model'
        validator._X_test = 'X_test'
        is_correct, msg = validator._step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.predict() should only be assigned to a single variable")

    def test_incorrect_args(self):
        cases = [
            {
                'desc': 'Empty kwargs with incorrect args',
                'content': 'y_pred = model.predict(y_train)'
            },
            {
                'desc': 'Empty args with incorrect kwargs',
                'content': 'y_pred = model.predict(X=y_train)'
            },
            {
                'desc': 'Non-empty args and non-empty kwargs',
                'content': 'y_pred = model.predict(X_test, X=X_test)'
            },
            {
                'desc': 'Empty args and empty kwargs',
                'content': 'y_pred = model.predict()'
            },
            {
                'desc': 'Incorrect number of args',
                'content': 'y_pred = model.predict(X_train, y_train)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['content'])
                validator._model = 'model'
                validator._X_test = 'X_test'
                is_correct, msg = validator._step_4_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'Incorrect parameters for model.predict(), are you correctly passing the test data to it?')


if __name__ == '__main__':
    unittest.main()
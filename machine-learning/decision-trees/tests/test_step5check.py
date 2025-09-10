from base_test import BaseTestValidator
import unittest


class TestStep5Check(BaseTestValidator):
    def test_success(self):
        cases = [
            {
                'desc': 'Passing parameters using args',
                'content': 'accuracy = accuracy_score(y_test, y_pred)'
            },
            {
                'desc': 'Passing parameters using kwargs',
                'content': 'accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['content'])
                validator._y_test = 'y_test'
                validator._y_pred = 'y_pred'
                is_correct, msg = validator._step_5_check()
                self.assertEqual(validator._accuracy, 'accuracy')
                self.assertTrue(is_correct)
                self.assertEqual(msg, '')

    def test_missing_function_call(self):
        content = ''
        validator = self.create_validator(content)
        is_correct, msg = validator._step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "accuracy_score() isn't called")
    
    def test_function_called_more_than_once(self):
        content = """
accuracy = accuracy_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
"""
        validator = self.create_validator(content)
        is_correct, msg = validator._step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "accuracy_score() shouldn't be called more than once")

    def test_output_not_assigned(self):
        content = 'accuracy_score(y_test, y_pred)'
        validator = self.create_validator(content)
        is_correct, msg = validator._step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "accuracy_score() should only be assigned to a single variable")

    def test_incorrect_args(self):
        cases = [
            {
                'desc': 'Empty kwargs with incorrect args',
                'content': 'accuracy = accuracy_score(y_train, y_test)'
            },
            {
                'desc': 'Empty args with incorrect kwargs',
                'content': 'accuracy = accuracy_score(y_true=y_train, y_pred=y_test)'
            },
            {
                'desc': 'Empty args and empty kwargs',
                'content': 'accuracy = accuracy_score()'
            },
            {
                'desc': 'Incorrect number of args',
                'content': 'accuracy = accuracy_score(y_true=y_test)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['content'])
                validator._y_test = 'y_test'
                validator._y_pred = 'y_pred'
                is_correct, msg = validator._step_5_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'Incorrect parameters for accuracy_score(), are you correctly passing the test output and model output to it?')


if __name__ == '__main__':
    unittest.main()
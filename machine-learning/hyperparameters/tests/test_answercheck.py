from base_test import BaseTestValidator
import unittest


class TestAnswerCheck(BaseTestValidator):
    def test_success(self):
        content = 'n_estimators = 140'
        validator = self.create_validator(content)
        is_correct, msg = validator._check_answer()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')

    def test_missing_variable(self):
        content = ''
        validator = self.create_validator(content)
        is_correct, msg = validator._check_answer()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Make sure you have a variable called 'n_estimators'")

    def test_incorrect_variable_type(self):
        cases = [
            {
                'desc': 'String variable',
                'content': "n_estimators = 'test'"
            },
            {
                'desc': 'Float variable',
                'content': "n_estimators = 123.45"
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['content'])
                is_correct, msg = validator._check_answer()
                self.assertFalse(is_correct)
                self.assertEqual(msg, "Make sure you're assigning an integer value to 'n_estimators'")

    def test_answer_too_low(self):
        content = "n_estimators = 100"
        validator = self.create_validator(content)
        is_correct, msg = validator._check_answer()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Incorrect (HINT: Higher 'n_estimators' value)")

    def test_answer_too_high(self):
        content = "n_estimators = 150"
        validator = self.create_validator(content)
        is_correct, msg = validator._check_answer()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Incorrect (HINT: Lower 'n_estimators' value)")


if __name__ == '__main__':
    unittest.main()
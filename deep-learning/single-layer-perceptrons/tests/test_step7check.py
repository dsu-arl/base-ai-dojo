from base_test import BaseTestValidator
import unittest


class TestStep7Check(BaseTestValidator):
    def test_success(self):
        script_content = """
rounded_preds = predictions.round()
"""
        validator = self.create_validator(script_content)
        validator.predictions = 'predictions'
        is_correct, msg = validator._step_7_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
        self.assertEqual(validator.predictions, 'rounded_preds')
    
    def test_round_not_called(self):
        script_content = ''
        validator = self.create_validator(script_content)
        validator.predictions = 'predictions'
        is_correct, msg = validator._step_7_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "predictions.round() isn't called")
    
    def test_round_called_more_than_once(self):
        script_content = """
predictions1 = predictions.round()
predictions2 = predictions.round()
"""
        validator = self.create_validator(script_content)
        validator.predictions = 'predictions'
        is_correct, msg = validator._step_7_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "predictions.round() should only be called once")

    def test_incorrect_args_kwargs(self):
        script_content = """
predictions = predictions.round(23, test_arg='test')
"""
        validator = self.create_validator(script_content)
        validator.predictions = 'predictions'
        is_correct, msg = validator._step_7_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "predictions.round() shouldn't have any parameters passed to it for this challenge")


if __name__ == '__main__':
    unittest.main()

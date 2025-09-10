from base_test import BaseTestValidator
import unittest


class TestStep6Check(BaseTestValidator):
    def test_success(self):
        script_content = """
predictions = model.predict(X)
"""
        validator = self.create_validator(script_content)
        validator.X = 'X'
        validator.model = 'model'
        is_correct, msg = validator._step_6_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
        self.assertEqual(validator.predictions, 'predictions')
    
    def test_predict_not_called(self):
        script_content = ''
        validator = self.create_validator(script_content)
        validator.model = 'model'
        is_correct, msg = validator._step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.predict() isn't called")
    
    def test_predict_called_more_than_once(self):
        script_content = """
predictions1 = model.predict(X)
predictions2 = model.predict(X)
"""
        validator = self.create_validator(script_content)
        validator.model = 'model'
        is_correct, msg = validator._step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.predict() should only be called once")
    
    def test_output_not_assigned(self):
        script_content = """
model.predict(X)
"""
        validator = self.create_validator(script_content)
        validator.model = 'model'
        is_correct, msg = validator._step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'model.predict() output should be assigned to a variable')

    def test_incorrect_args(self):
        script_content = """
predictions = model.predict(X, y)
"""
        validator = self.create_validator(script_content)
        validator.model = 'model'
        validator.X = 'X'
        validator.y = 'y'
        is_correct, msg = validator._step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'model.predict() should take a single argument X')
    
    def test_incorrect_kwargs(self):
        script_content = """
predictions = model.predict(X, test_arg='test')
"""
        validator = self.create_validator(script_content)
        validator.model = 'model'
        validator.X = 'X'
        is_correct, msg = validator._step_6_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'model.predict() should not have any keyword arguments for this challenge')


if __name__ == '__main__':
    unittest.main()

from base_test import BaseTestValidator
import unittest


class TestStep5Check(BaseTestValidator):
    def test_success(self):
        script_content = """
model.fit(X, y, epochs=100, verbose=1)
"""
        validator = self.create_validator(script_content)
        validator.X = 'X'
        validator.y = 'y'
        validator.model = 'model'
        is_correct, msg = validator._step_5_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
    
    def test_fit_not_called(self):
        script_content = ''
        validator = self.create_validator(script_content)
        validator.model = 'model'
        is_correct, msg = validator._step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.fit() isn't called")
    
    def test_fit_called_more_than_once(self):
        script_content = """
model.fit(X, y, epochs=100, verbose=1)
model.fit(X, y, epochs=100, verbose=1)
"""
        validator = self.create_validator(script_content)
        validator.model = 'model'
        is_correct, msg = validator._step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.fit() should only be called once")
    
    def test_fit_output_assigned_to_variable(self):
        script_content = """
test = model.fit(X, y, epochs=100, verbose=1)
"""
        validator = self.create_validator(script_content)
        validator.model = 'model'
        is_correct, msg = validator._step_5_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.fit() shouldn't be assigned to a variable")
    
    def test_incorrect_fit_args(self):
        cases = [
            {
                'desc': 'Missing X parameter',
                'script_content': 'model.fit([], y, epochs=100, verbose=1)'
            },
            {
                'desc': 'Missing y parameter',
                'script_content': 'model.fit(X, [], epochs=100, verbose=1)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['script_content'])
                validator.X = 'X'
                validator.y = 'y'
                validator.model = 'model'
                is_correct, msg = validator._step_5_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'Missing or incorrect parameters, are you passing your dataset and labels to model.fit()?')
    
    def test_incorrect_fit_kwargs(self):
        cases = [
            {
                'desc': 'Missing epochs parameter',
                'script_content': 'model.fit(X, y, verbose=1)'
            },
            {
                'desc': 'Incorrect epochs parameter',
                'script_content': 'model.fit(X, y, epochs=50, verbose=1)'
            },
            {
                'desc': 'Missing verbose parameter',
                'script_content': 'model.fit(X, y, epochs=100)'
            },
            {
                'desc': 'Incorrect verbose parameter',
                'script_content': 'model.fit(X, y, epochs=100, verbose=0)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['script_content'])
                validator.X = 'X'
                validator.y = 'y'
                validator.model = 'model'
                is_correct, msg = validator._step_5_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'Missing or incorrect parameters, are you training for 100 epochs and outputting the training output?')


if __name__ == '__main__':
    unittest.main()

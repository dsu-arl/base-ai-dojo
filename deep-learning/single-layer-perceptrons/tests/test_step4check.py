from base_test import BaseTestValidator
import unittest


class TestStep4Check(BaseTestValidator):
    def test_success(self):
        script_content = """
model.compile(optimizer='adam', loss='binary_crossentropy')
"""
        validator = self.create_validator(script_content)
        validator.model = 'model'
        is_correct, msg = validator._step_4_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
    
    def test_compile_not_called(self):
        script_content = ''
        validator = self.create_validator(script_content)
        validator.model = 'model'
        is_correct, msg = validator._step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.compile() isn't called")
    
    def test_compile_called_more_than_once(self):
        script_content = """
model.compile(optimizer='adam', loss='binary_crossentropy')
model.compile(optimizer='adam', loss='binary_crossentropy')
"""
        validator = self.create_validator(script_content)
        validator.model = 'model'
        is_correct, msg = validator._step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.compile() should only be called once")

    def test_compile_output_assigned_to_variable(self):
        script_content = """
test = model.compile(optimizer='adam', loss='binary_crossentropy')
"""
        validator = self.create_validator(script_content)
        validator.model = 'model'
        is_correct, msg = validator._step_4_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "model.compile() shouldn't be assigned to a variable")
    
    def test_incorrect_compile_params(self):
        cases = [
            {
                'desc': 'incorrect optimizer',
                'script_content': """
model.compile(optimizer='', loss='binary_crossentropy')
"""
            },
            {
                'desc': 'incorrect loss function',
                'script_content': """
model.compile(optimizer='adam', loss='')
"""
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['script_content'])
                validator.model = 'model'
                is_correct, msg = validator._step_4_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, "Missing or incorrect optimizer and loss function passed to model.compile()")


if __name__ == '__main__':
    unittest.main()

from base_test import BaseTestValidator
import unittest


class TestCheckY(BaseTestValidator):
    def test_success(self):
        script_content = 'y = np.random.randint(0, 2, size=500)'
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_y()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
        self.assertEqual(validator.y, 'y')
    
    def test_rand_not_called(self):
        script_content = ''
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_y()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "np.random.randint() isn't called")
    
    def test_rand_called_more_than_once(self):
        script_content = """
y = np.random.randint(0, 2, size=500)
y = np.random.randint(0, 2, size=500)
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_y()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "np.random.randint() should only be called once")
    
    def test_output_not_assigned_to_variable(self):
        script_content = 'np.random.randint(0, 5, size=500)'
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_y()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Make sure you're storing the output of np.random.randint() in a variable")

    def test_incorrect_args(self):
        cases = [
            {
                'desc': 'Incorrect low parameter value',
                'script_content': 'y = np.random.randint(1, 2, size=500)'
            },
            {
                'desc': 'Incorrect high parameter value',
                'script_content': 'y = np.random.randint(0, 5, size=500)'
            },
            {
                'desc': 'Incorrect size parameter value',
                'script_content': 'y = np.random.randint(0, 2, size=100)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['script_content'])
                is_correct, msg = validator.check_y()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'Did you pass the correct values in the instructions to np.random.randint()?')


if __name__ == '__main__':
    unittest.main()

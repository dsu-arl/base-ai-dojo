from base_test import BaseTestValidator
import unittest


class TestCheckX(BaseTestValidator):
    def test_success(self):
        script_content = 'X = np.random.rand(500, 7)'
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_X()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
        self.assertEqual(validator.X, 'X')
    
    def test_rand_not_called(self):
        script_content = ''
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_X()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "np.random.rand() isn't called")
    
    def test_rand_called_more_than_once(self):
        script_content = """
X = np.random.rand(500, 7)
X = np.random.rand(500, 7)
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_X()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "np.random.rand() should only be called once")
    
    def test_output_not_assigned_to_variable(self):
        script_content = 'np.random.rand(500, 7)'
        validator = self.create_validator(script_content)
        is_correct, msg = validator.check_X()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Make sure you're storing the output of np.random.rand() in a variable")

    def test_incorrect_args(self):
        cases = [
            {
                'desc': 'Incorrect number of samples',
                'script_content': 'X = np.random.rand(5, 3)'
            },
            {
                'desc': 'Incorrect number of features',
                'script_content': 'X = np.random.rand(500, 30)'
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['script_content'])
                is_correct, msg = validator.check_X()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'Did you pass the correct values in the instructions to np.random.rand()?')


if __name__ == '__main__':
    unittest.main()

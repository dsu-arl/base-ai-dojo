from base_test import BaseTestValidator
import unittest


class TestStep2Check(BaseTestValidator):
    def test_success(self):
        script_content = """
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_2_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
        self.assertEqual(validator.X, 'X')
        self.assertEqual(validator.y, 'y')
    
    def test_missing_nparray(self):
        script_content = """
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [1]]
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "np.array isn't called")
    
    def test_nparray_called_more_than_twice(self):
        script_content = """
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'np.array should only be called twice, once for X and once for y')
    
    def test_assignment_to_variable(self):
        cases = [
            {
                'desc': 'X not assigned, y assigned',
                'script_content': """
np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])
"""
                },
                {
                "desc": "X assigned, y not assigned",
                "script_content": """
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
np.array([[0], [1], [1], [1]])
"""
            },
            {
                "desc": "Neither assigned",
            "script_content": """
import numpy as np

np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
np.array([[0], [1], [1], [1]])
"""
            }
        ]
        
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['script_content'])
                is_correct, msg = validator._step_2_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, 'np.array should be assigned to a variable')
    
    def test_incorrect_X_nparray_args(self):
        script_content = """
X = np.array([])
y = np.array([[0], [1], [1], [1]])
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Data passed to X doesn't match instructions.")
    
    def test_incorrect_y_nparray_args(self):
        script_content = """
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([])
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_2_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "Data passed to y doesn't match instructions.")


if __name__ == '__main__':
    unittest.main()

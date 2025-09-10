from base_test import BaseTestValidator
import unittest


class TestStep3Check(BaseTestValidator):
    def test_success(self):
        script_content = """
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_3_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
        self.assertEqual(validator.model, 'model')

    def test_function_not_called(self):
        script_content = ''
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "tf.keras.Sequential isn't called")
    
    def test_function_more_than_once(self):
        script_content = """
model = tf.keras.Sequential()
model2 = tf.keras.Sequential()
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "tf.keras.Sequential should only be called once")
    
    def test_output_not_assigned(self):
        script_content = """
tf.keras.Sequential()
"""
        validator = self.create_validator(script_content)
        is_correct, msg = validator._step_3_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, "tf.keras.Sequential output should be assigned to a variable")
   
    def test_incorrect_model_args(self):
        cases = [
            {
                'desc': 'missing Input layer',
                'script_content': """
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
"""
            },
            {
                'desc': 'missing Dense layer parameters',
                'script_content': """
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense()
])
"""
            },
            {
                'desc': 'incorrect units parameter value',
                'script_content': """
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(units=5, activation='sigmoid')
])
"""
            },
            {
                'desc': 'incorrect input shape parameter value',
                'script_content': """
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
"""
            },
            {
                'desc': 'incorrect activation parameter value',
                'script_content': """
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(units=1, activation='relu')
])
"""
            }
        ]
        for case in cases:
            with self.subTest(case=case['desc']):
                validator = self.create_validator(case['script_content'])
                is_correct, msg = validator._step_3_check()
                self.assertFalse(is_correct)
                self.assertEqual(msg, "Missing or incorrect layers for perceptron")


if __name__ == '__main__':
    unittest.main()

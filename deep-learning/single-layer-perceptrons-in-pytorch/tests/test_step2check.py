from base_test import BaseTestValidator
import unittest


class TestStep2Check(BaseTestValidator):
    def test_success(self):
        content = """
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
"""
        validator = self.create_validator(content)
        is_correct, msg = validator._step_2_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
    
    def test_incorrect_init_function(self):
        pass

    def test_incorrect_forward_function(self):
        pass


if __name__ == '__main__':
    unittest.main()

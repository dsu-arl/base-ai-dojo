from base_test import BaseTestValidator
import unittest


class TestStep1Check(BaseTestValidator):
    def test_success(self):
        content = """
import torch
import torch.nn as nn
import numpy as np
"""
        validator = self.create_validator(content)
        is_correct, msg = validator._step_1_check()
        self.assertTrue(is_correct)
        self.assertEqual(msg, '')
    
    def test_missing_torch_import(self):
        content = """
import torch.nn as nn
import numpy as np
"""
        validator = self.create_validator(content)
        is_correct, msg = validator._step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'Missing torch import')

    def test_missing_torchnn_import(self):
        content = """
import torch
import numpy as np
"""
        validator = self.create_validator(content)
        is_correct, msg = validator._step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'Missing torch.nn import')

    def test_incorrect_torchnn_alias(self):
        content = """
import torch
import torch.nn as tn
import numpy as np
"""
        validator = self.create_validator(content)
        is_correct, msg = validator._step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'Incorrect alias for torch.nn import')

    def test_missing_numpy_import(self):
        content = """
import torch
import torch.nn as nn
"""
        validator = self.create_validator(content)
        is_correct, msg = validator._step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'Missing numpy import')

    def test_incorrect_numpy_alias(self):
        content = """
import torch
import torch.nn as nn
import numpy as nu
"""
        validator = self.create_validator(content)
        is_correct, msg = validator._step_1_check()
        self.assertFalse(is_correct)
        self.assertEqual(msg, 'Incorrect alias for numpy import')


if __name__ == '__main__':
    unittest.main()

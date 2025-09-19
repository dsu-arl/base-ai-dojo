#!/usr/bin/exec-suid -- /usr/bin/python3.12 -I
import sys
sys.path.append('/challenge')

def print_flag():
    try:
        with open("/flag", "r") as f:
            print(f.read())
    except FileNotFoundError:
        print("Error: Flag file not found.")

# Add your imports and other code below here
from paceAITester.datatypes import FunctionCall
from paceAITester.verify_helpers import *
from typing import Tuple


class Validator:
    def __init__(self, script_path: str):
        self.lines = extract_python_details(script_path)
        self.script_path = script_path

        # User variables
        self.X = None
        self.y = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.outputs = None
        self.loss = None
        self.predictions = None

        self.checks = []
    
    def _step_1_check(self) -> Tuple[bool, str]:
        """Step Goal: Import the libraries torch, torch.nn with alias 'nn', and numpy with alias 'np'.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or failure of 
            the validation, and a string message providing error details if failure.
        """
        with open(self.script_path, 'r') as f:
            file_lines = f.read().split('\n')

        if 'import torch' not in file_lines:
            return False, 'Missing torch import'
        
        if not any('import torch.nn' in s for s in file_lines):
            return False, 'Missing torch.nn import'
        
        if 'import torch.nn as nn' not in file_lines:
            return False, 'Incorrect alias for torch.nn import'
        
        if not any('import numpy' in s for s in file_lines):
            return False, 'Missing numpy import'
        
        if 'import numpy as np' not in file_lines:
            return False, 'Incorrect alias for numpy import'
        
        return True, ''

    def _step_2_check(self) -> Tuple[bool, str]:
        # Checks if perceptron class is correctly defined
        """Step Goal: Build a single-layer perceptron class called Perceptron for model 
        using `nn.Module`

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or failure of 
            the validation, and a string message providing error details if failure.
        """
        print(self.lines)

    def _step_3_check(self) -> Tuple[bool, str]:
        # Checks if sample data is correctly defined
        pass

    def _step_4_check(self) -> Tuple[bool, str]:
        # Checks if sample data is correctly converted to PyTorch tensors
        pass

    def _step_5_check(self) -> Tuple[bool, str]:
        # Checks if perceptron model is correctly instantiated
        pass

    def _step_6_check(self) -> Tuple[bool, str]:
        # Checks if loss function and optimizer are correctly defined
        pass

    def _step_7_check(self) -> Tuple[bool, str]:
        # Checks if training loop is correctly implemented
        # Forward pass, clear previous gradients, compute gradients, update weights
        pass

    def _step_8_check(self) -> Tuple[bool, str]:
        # Make predictions using trained model and round them
        pass

    def _step_9_check(self) -> Tuple[bool, str]:
        # Checks if rounded predictions are converted back to NumPy array and printed out
        pass

    def verify_code(self) -> None:
        run_verification(self.checks)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: verify <user_script>.py')
        sys.exit(1)
    
    script_path = sys.argv[1]

    validator = Validator(script_path)
    validator.verify_code()
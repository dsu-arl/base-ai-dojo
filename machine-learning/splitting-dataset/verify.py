#!/usr/bin/exec-suid -- /usr/bin/python3.12 -I
import sys
sys.path.append('/challenge')

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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.checks = [self.check_X, self.check_y, self.check_split]
    
    def check_X(self) -> Tuple[bool, str]:
        """
        Step Goal: Create X variable using np.random.rand() which has 500 samples and 7 features.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = 'np.random.rand'
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name}() isn't called"
        if len(function_calls) > 1:
            return False, f"{function_name}() should only be called once"
        
        function_call = FunctionCall.from_dict(function_calls[0])

        if function_call.variable is None:
            return False, f"Make sure you're storing the output of {function_name}() in a variable"
    
        self.X = function_call.variable

        if function_call.args != ['500', '7']:
            return False, f'Did you pass the correct values in the instructions to {function_name}()?'
        
        return True, ''

    def check_y(self) -> Tuple[bool, str]:
        """
        Step Goal: Create y using np.random.randint() which has the same amount of samples and outputs either 0 or 1.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = 'np.random.randint'
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name}() isn't called"
        if len(function_calls) > 1:
            return False, f"{function_name}() should only be called once"
        
        function_call = FunctionCall.from_dict(function_calls[0])

        if function_call.variable is None:
            return False, f"Make sure you're storing the output of {function_name}() in a variable"
    
        self.y = function_call.variable

        if function_call.args != ['0', '2'] or function_call.kwargs != {'size': '500'}:
            return False, f'Did you pass the correct values in the instructions to {function_name}()?'

        return True, ''

    def check_split(self) -> Tuple[bool, str]:
        """
        Step Goal: Split the dataset where the train dataset is 70% of the original dataset.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = 'train_test_split'
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name}() isn't called"
        if len(function_calls) > 1:
            return False, f'{function_name}() should only be called once'
        
        function_call = FunctionCall.from_dict(function_calls[0])

        if function_call.variable is None:
            return False, "Make sure you're storing the output of train_test_split() in variables for your train and test datasets (HINT: It's 4 variables)"
        
        if len(function_call.variable) != 4:
            return False, 'Incorrect number of output variables.'

        self.X_train = function_call.variable[0]
        self.X_test = function_call.variable[1]
        self.y_train = function_call.variable[2]
        self.y_test = function_call.variable[3]

        if function_call.args != [self.X, self.y]:
            return False, 'Did you provide the correct variables for X and y?'
    
        if function_call.kwargs != {'test_size': '0.3'}:
            return False, "Did you correctly set the 'test_size' parameter so that the training dataset is 70% of the original dataset?"

        return True, ''

    def verify_code(self) -> None:
        run_verification(self.checks)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: verify <user_script>.py')
        sys.exit(1)
    
    script_path = sys.argv[1]

    validator = Validator(script_path)
    validator.verify_code()

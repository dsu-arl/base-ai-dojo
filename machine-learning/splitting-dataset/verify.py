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
        return True, ''

    def check_split(self) -> Tuple[bool, str]:
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

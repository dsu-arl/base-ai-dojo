#!/usr/bin/exec-suid -- /usr/bin/python3.12 -I
import sys
sys.path.append('/challenge')

from paceAITester.datatypes import FunctionCall
from paceAITester.verify_helpers import *
from typing import Tuple


class Validator:
    def __init__(self, script_path: str):
        self._lines = extract_python_details(script_path)
        self._script_path = script_path

        # User variables
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None
        self._model = None
        self._y_pred = None
        self._mse = None

        self._checks = [
            self._step_1_check, self._step_2_check, self._step_3_check,
            self._step_4_check, self._step_5_check, self._step_6_check
        ]

    def _step_1_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Split the dataset where the test dataset is 20% of the original dataset with a random state of 42.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = 'train_test_split'
        function_calls = find_function_call(self._lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name}() isn't called"
        if len(function_calls) != 1:
            return False, f'{function_name}() should only be called once'
        
        function_call = FunctionCall.from_dict(function_calls[0])

        if len(function_call.variable) != 4:
            return False, 'train_test_split() should be unpacked into 4 variables'
    
        self._X_train, self._X_test, self._y_train, self._y_test = function_call.variable
        
        if function_call.args != ['X', 'y']:
            return False, 'You need to pass X and y to train_test_split() so knows what data to split'
        
        solution_kwargs = {'test_size': '0.2', 'random_state': '42'}
        if function_call.kwargs != solution_kwargs:
            return False, "train_test_split() parameters don't match what's expected, did you give 'test_size' and 'random_state' the correct values from the instructions?"

        return True, ''

    def verify_code(self) -> None:
        run_verification(self._checks)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: verify <user_script>.py')
        sys.exit(1)
    
    script_path = sys.argv[1]

    validator = Validator(script_path)
    validator.verify_code()
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
        self.model = None
        self.predictions = None

        self.checks = [
            self._step_1_check, self._step_2_check,
            self._step_3_check, self._step_4_check,
            self._step_5_check, self._step_6_check,
            self._step_7_check, self._step_8_check
        ]

    def _step_1_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Import tensorflow with the alias 'tf' and numpy with the alias 'np'.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        with open(self.script_path, 'r') as f:
            file_lines = f.read().split('\n')

        if 'import tensorflow as tf' not in file_lines:
            return False, 'Missing or incorrect tensorflow import line, did you import it with the specified alias?'
        
        if 'import numpy as np' not in file_lines:
            return False, 'Missing or incorrect numpy import line, did you import it with the specified alias?'
    
        return True, ''

    def _step_2_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Define dataset for OR operation.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = 'np.array'
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name} isn't called"
        
        if len(function_calls) != 2:
            return False, f'{function_name} should only be called twice, once for X and once for y'
        
        func_call_for_X = FunctionCall.from_dict(function_calls[0])
        func_call_for_y = FunctionCall.from_dict(function_calls[1])
        
        if func_call_for_X.variable is None or func_call_for_y.variable is None:
            return False, f'{function_name} should be assigned to a variable'
        
        self.X = func_call_for_X.variable
        self.y = func_call_for_y.variable

        if func_call_for_X.args != ['[[0, 0], [0, 1], [1, 0], [1, 1]]']:
            return False, f"Data passed to {self.X} doesn't match instructions."
        
        if func_call_for_y.args != ['[[0], [1], [1], [1]]']:
            return False, f"Data passed to {self.y} doesn't match instructions."

        return True, ''

    def _step_3_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Create single-layer perceptron with one dense layer, input shape of 2, sigmoid activation.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = 'tf.keras.Sequential'
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name} isn't called"
        if len(function_calls) > 1:
            return False, f"{function_name} should only be called once"
        
        function_call = FunctionCall.from_dict(function_calls[0])
        if output_not_assigned_to_variable(function_call):
            return False, f'{function_name} output should be assigned to a variable'

        self.model = function_call.variable

        # Make sure that args only contains Dense layer with correct arguments
        solution_args = ["[tf.keras.layers.Dense(units=1, input_shape=[2], activation='sigmoid')]"]
        if function_call.args != solution_args:
            return False, "Missing or incorrect parameters for Dense layer or model doesn't contain only a single Dense layer"

        return True, ''

    def _step_4_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Compile the model with adam optimzier and binary_crossentropy loss function.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = f'{self.model}.compile'
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name}() isn't called"
        if len(function_calls) > 1:
            return False, f"{function_name}() should only be called once"
    
        function_call = FunctionCall.from_dict(function_calls[0])
        if function_call.variable is not None:
            return False, f"{function_name}() shouldn't be assigned to a variable"
        
        solution_kwargs = {'optimizer': "'adam'", 'loss': "'binary_crossentropy'"}
        if function_call.kwargs != solution_kwargs:
            return False, f"Missing or incorrect optimizer and loss function passed to {function_name}()"

        return True, ''

    def _step_5_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Train the model for 1000 epochs.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = f'{self.model}.fit'
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name}() isn't called"
        if len(function_calls) > 1:
            return False, f"{function_name}() should only be called once"
        
        function_call = FunctionCall.from_dict(function_calls[0])
        if function_call.variable is not None:
            return False, f"{function_name}() shouldn't be assigned to a variable"

        solution_args = [self.X, self.y]
        if function_call.args != solution_args:
            return False, f"Missing or incorrect parameters, are you passing your dataset and labels to {function_name}()?"

        solution_kwargs = {'epochs': '1000', 'verbose': '0'}
        if function_call.kwargs != solution_kwargs:
            return False, f"Missing or incorrect parameters, are you training for 1000 epochs and suppressing the training output?"

        return True, ''

    def _step_6_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Use the trained model to make predictions for X, rounding predictions to 0 or 1.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        # Validation for model.predict()
        predict_function_name = f'{self.model}.predict'
        function_calls = find_function_call(self.lines, predict_function_name)

        if function_not_called(function_calls):
            return False, f"{predict_function_name}() isn't called"
        if len(function_calls) > 1:
            return False, f"{predict_function_name}() should only be called once"
        
        function_call = FunctionCall.from_dict(function_calls[0])
        if output_not_assigned_to_variable(function_call):
            return False, f'{predict_function_name}() output should be assigned to a variable'

        self.predictions = function_call.variable

        solution_args = [self.X]
        if function_call.args != solution_args:
            return False, f'{predict_function_name}() should take a single argument {self.X}'
        
        solution_kwargs = {}
        if function_call.kwargs != solution_kwargs:
            return False, f'{predict_function_name}() should not have any keyword arguments for this challenge'

        return True, ''

    def _step_7_check(self) -> Tuple[bool, str]:
        # Validation for predictions.round()
        function_name = f'{self.predictions}.round'
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name}() isn't called"
        if len(function_calls) > 1:
            return False, f"{function_name}() should only be called once"
        
        function_call = FunctionCall.from_dict(function_calls[0])

        if function_call.args != [] or function_call.kwargs != {}:
            return False, f"{function_name}() shouldn't have any parameters passed to it for this challenge"

        # Reassign self.predictions if user stored result in new variable after rounding
        if function_call.variable is not None and function_call.variable != self.predictions:
            self.predictions = function_call.variable
        
        return True, ''

    def _step_8_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Print the rounded predictions from the trained model.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = 'print'
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name}() isn't called"
        if len(function_calls) > 1:
            return False, f"{function_name}() should only be called once"
        
        function_call = FunctionCall.from_dict(function_calls[0])

        solution_args = [self.predictions]
        if function_call.args != solution_args:
            return False, f'Incorrect parameters passed to {function_name}(), are you passing your rounded predictions to the {function_name}() function?'

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

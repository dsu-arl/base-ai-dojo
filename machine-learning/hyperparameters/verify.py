#!/usr/bin/exec-suid -- /usr/bin/python3.12 -I
import sys
sys.path.append('/challenge')

from paceAITester.verify_helpers import *
from typing import Dict, Optional, Tuple


class Validator:
    def __init__(self, script_path: str):
        self._script_path = script_path
        self._lines = extract_python_details(script_path)
        self._variables = retrieve_variable_values(script_path)

    def _find_variable(self, variable_name: str) -> Optional[Dict[str, str]]:
        for key, value in self._variables.items():
            if key == variable_name:
                return {'name': key, 'value': value}

        return None

    def _check_template(self) -> Tuple[bool, str]:
        """
        Step Goal: Ensure that the template code is correctly used in Python file.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        correct_template = [
            {'variable': 'iris', 'function': 'load_iris', 'args': [], 'kwargs': {}},
            {'variable': ('X_train', 'X_test', 'y_train', 'y_test'), 'function': 'train_test_split', 'args': ['X', 'y'], 'kwargs': {'test_size': '0.2', 'random_state': '42'}},
            {'variable': 'model', 'function': 'RandomForestClassifier', 'args': [], 'kwargs': {'n_estimators': 'n_estimators', 'random_state': '42'}},
            {'variable': None, 'function': 'model.fit', 'args': ['X_train', 'y_train'], 'kwargs': {}},
            {'variable': 'y_pred_proba', 'function': 'model.predict_proba', 'args': ['X_test'], 'kwargs': {}},
            {'variable': 'loss', 'function': 'log_loss', 'args': ['y_test', 'y_pred_proba'], 'kwargs': {}},
            {'variable': None, 'function': 'print', 'args': ["'Target Loss: 0.0252'"], 'kwargs': {}},
            {'variable': None, 'function': 'print', 'args': ["f' Model Loss: {loss:.4f}'"], 'kwargs': {}}
        ]

        if self._lines != correct_template:
            return False, "No need to modify the given code template, only modify the 'n_estimators' value"

        return True, ''
    
    def _check_answer(self) -> Tuple[bool, str]:
        """
        Step Goal: Ensure that the n_estimators variable has the correct value.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        variable_call = self._find_variable('n_estimators')
        if variable_call is None:
            return False, "Make sure you have a variable called 'n_estimators'"
        
        if variable_call['value'] == 'Unresolvable dynamic value' or not isinstance(variable_call['value'], int):
            return False, "Make sure you're assigning an integer value to 'n_estimators'"
        
        guess = variable_call['value']
        if guess != 140:
            if guess > 140:
                direction = 'Lower'
            else:
                direction = 'Higher'
            return False, f"Incorrect (HINT: {direction} 'n_estimators' value)"

        return True, ''

    def verify_code(self) -> None:
        red_text_code = '\033[31m'
        green_text_code = '\033[32m'
        reset_code = '\033[0m'

        template_correct, error_msg = self._check_template()
        if not template_correct:
            print(f"{red_text_code}{error_msg}{reset_code}")
            sys.exit(1)
        
        passed, error_msg = self._check_answer()
        if not passed:
            print(f"{red_text_code}{error_msg}{reset_code}")
            sys.exit(1)
    
        print(f'{green_text_code}Correct{reset_code}')
        print('Congratulations! You have passed this challenge! Here is your flag:')
        print_flag()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: verify <user_script>.py')
        sys.exit(1)
    
    script_path = sys.argv[1]

    validator = Validator(script_path)
    validator.verify_code()
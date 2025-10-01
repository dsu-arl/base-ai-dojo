"""Verification script for Splitting Datasets challenge."""

#!/usr/bin/exec-suid -- /usr/bin/python3.12 -I
import sys

sys.path.append("/challenge")

from dataclasses import dataclass
from typing import Any, Optional, Tuple
from paceAITester.datatypes import AssignStatement, FunctionCallStatement
from paceAITester.parser import StatementParser
from paceAITester.utils import (
    find_function_calls,
    function_not_called,
    run_verification,
)


@dataclass
class UserVariables:
    """Stores the names of user variables"""

    x: Optional[Any] = None
    y: Optional[Any] = None
    x_train: Optional[Any] = None
    x_test: Optional[Any] = None
    y_train: Optional[Any] = None
    y_test: Optional[Any] = None


class Validator:
    """Validator class that runs verification checks on user submitted code for
    challenge.
    """

    def __init__(self, script_path: str):
        self.parser = StatementParser(script_path)
        self.lines = self.parser.parse()
        self.user_vars = UserVariables()
        self.checks = [self.check_x, self.check_y, self.check_split]

    def check_x(self) -> Tuple[bool, str]:
        """Step Goal: Create X variable using np.random.rand() which has 500 samples and
        7 features.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = "np.random.rand"
        function_calls = find_function_calls(self.lines, function_name)

        error_msg = None
        if function_not_called(function_calls):
            error_msg = f"{function_name}() isn't called"
        elif len(function_calls) > 1:
            error_msg = f"{function_name}() should only be called once"
        elif not isinstance(function_calls[0], AssignStatement):
            error_msg = (
                "Make sure you're storing the output of "
                + f"{function_name}() in a variable"
            )
        else:
            function_call: AssignStatement = function_calls[0]

            if len(function_call.targets) != 1:
                error_msg = (
                    f"The output of {function_name}() should only be stored in a "
                    + "single variable"
                )
            elif function_call.value.args != ["500", "7"]:
                error_msg = (
                    "Did you pass the correct values in the instructions "
                    + f"to {function_name}()?"
                )

        if error_msg:
            return False, error_msg

        self.user_vars.x = function_call.targets[0]

        return True, ""

    def check_y(self) -> Tuple[bool, str]:
        """Step Goal: Create y using np.random.randint() which has the same amount of
        samples and outputs either 0 or 1.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = "np.random.randint"
        statements = find_function_calls(self.lines, function_name)

        error_msg = None
        if function_not_called(statements):
            error_msg = f"{function_name}() isn't called"
        elif len(statements) > 1:
            error_msg = f"{function_name}() should only be called once"
        elif not isinstance(statements[0], AssignStatement):
            error_msg = (
                "Make sure you're storing the output of "
                + f"{function_name}() in a variable"
            )
        else:
            statement: AssignStatement = statements[0]
            if isinstance(statement.value, FunctionCallStatement):
                function_call: FunctionCallStatement = statement.value
            else:
                raise TypeError(
                    "Expected FunctionCallStatement, "
                    + f"got {type(statement.value).__name__}"
                )

            if len(statement.targets) != 1:
                error_msg = (
                    f"The output of {function_name}() should only be stored in a "
                    + "single variable"
                )
            elif function_call.args != ["0", "2"] or function_call.kwargs != {
                "size": "500"
            }:
                error_msg = (
                    "Did you pass the correct values in the instructions "
                    + f"to {function_name}()?"
                )

        if error_msg:
            return False, error_msg

        self.user_vars.y = statement.targets[0]

        return True, ""

    def check_split(self) -> Tuple[bool, str]:
        """Step Goal: Split the dataset where the train dataset is 70% of the original
        dataset.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = "train_test_split"
        statements = find_function_calls(self.lines, function_name)

        error_msg = None
        if function_not_called(statements):
            error_msg = f"{function_name}() isn't called"
        elif len(statements) > 1:
            error_msg = f"{function_name}() should only be called once"
        elif not isinstance(statements[0], AssignStatement):
            error_msg = (
                "Make sure you're storing the output of train_test_split() "
                + "in variables for your train and test datasets "
                + "(HINT: It's 4 variables)"
            )
        else:
            function_call = statements[0]
            statement: AssignStatement = statements[0]
            if isinstance(statement.value, FunctionCallStatement):
                function_call: FunctionCallStatement = statement.value
            else:
                raise TypeError(
                    "Expected FunctionCallStatement, "
                    + f"got {type(statement.value).__name__}"
                )

            if len(statement.targets) != 4:
                error_msg = "Incorrect number of output variables."
            elif function_call.args != [self.user_vars.x, self.user_vars.y]:
                error_msg = "Did you provide the correct variables for X and y?"
            elif function_call.kwargs != {"test_size": "0.3"}:
                error_msg = (
                    "Did you correctly set the 'test_size' parameter so that the "
                    + "training dataset is 70% of the original dataset?"
                )

        if error_msg:
            return False, error_msg

        self.user_vars.x_train = statement.targets[0]
        self.user_vars.x_test = statement.targets[1]
        self.user_vars.y_train = statement.targets[2]
        self.user_vars.y_test = statement.targets[3]

        return True, ""

    def verify_code(self) -> None:
        """Runs the list of verification functions to make sure the user submitted code
        is correct.
        """
        run_verification(self.checks)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: verify <user_script>.py")
        sys.exit(1)

    filepath = sys.argv[1]

    validator = Validator(filepath)
    validator.verify_code()

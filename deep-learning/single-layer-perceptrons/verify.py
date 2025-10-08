#!/usr/bin/exec-suid -- /usr/bin/python3.12 -I
import sys

sys.path.append("/challenge")

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from paceAITester.datatypes import AssignStatement, FunctionCallStatement, ImportStatement, ImportFromStatement, Statement
from paceAITester.parser import StatementParser
from paceAITester.utils import (
    find_function_calls,
    run_verification,
)


@dataclass
class UserVariables:
    """Stores the names of user variables"""

    x: Optional[Any] = None
    y: Optional[Any] = None
    model: Optional[Any] = None
    predictions: Optional[Any] = None


class Validator:
    """Validator class that runs verification checks on user submitted code for
    challenge.
    """

    def __init__(self, script_path: str):
        self.parser = StatementParser(script_path)
        self.lines = self.parser.parse()
        self.user_vars = UserVariables()
        self.checks = [
            self.step_1_check,
            self.step_2_check,
            self.step_3_check,
            self.step_4_check,
            self.step_5_check,
            self.step_6_check,
            self.step_7_check,
            self.step_8_check,
        ]

    def _validate_basic_call(
        self, function_name: str, should_assign: Optional[bool] = None
    ) -> Tuple[Optional[str], List[Any]]:
        """Validates basic function call requirements.

        Args:
            function_name (str): Name of the function to validate.
            should_assign (Optional[bool], optional): If True, requires assignment;
                if False, forbids it; if None, doesn't check.

        Returns:
            Tuple[Optional[str], List[Any]]: Tuple of (error_message, statements). If
                error_message is None, validation passed
        """
        statements = find_function_calls(self.lines, function_name)

        if len(statements) == 0:
            return f"{function_name}() isn't called", statements

        if len(statements) != 1:
            return f"{function_name}() should only be called once", statements

        if should_assign is True and not isinstance(statements[0], AssignStatement):
            return (
                f"Make sure you store the output of {function_name}() in a variable",
                statements,
            )

        if should_assign is False and isinstance(statements[0], AssignStatement):
            return (
                f"{function_name}() shouldn't be assigned to any variables",
                statements,
            )

        return None, statements

    def _get_function_call(
        self, statement: Statement, allow_assign: bool = True
    ) -> FunctionCallStatement:
        """Extracts FunctionCallStatement from a statement.

        Args:
            statement (Statement): The statement to extract from.
            allow_assign (bool, optional): Whether to extract from AssignStatement.
                Defaults to True.

        Returns:
            FunctionCallStatement: Extracted function call statement.

        Raises:
            TypeError: If the statement type is unexpected.
        """
        if isinstance(statement, FunctionCallStatement):
            return statement

        if allow_assign and isinstance(statement, AssignStatement):
            if isinstance(statement.value, FunctionCallStatement):
                return statement.value

        raise TypeError(
            f"Expected FunctionCallStatement, got {type(statement).__name__}"
        )

    def _validate_args(
        self,
        function_call: FunctionCallStatement,
        solution_args: List[str],
        solution_kwargs: Dict[str, str],
    ) -> bool:
        """Validates function call arguments against expected solutions.

        Args:
            function_call (FunctionCallStatement): Function call statement to validate.
            solution_args (List[str]): Positional arguments solution.
            solution_kwargs (Dict[str, str]): Keyword arguments solution.

        Returns:
            bool: True if function call statement arguments match solution, else False.
        """
        # Supplies data using positional arguments
        if not function_call.kwargs and function_call.args != solution_args:
            return False

        # Supplies data using keyword arguments
        if not function_call.args and function_call.kwargs != solution_kwargs:
            return False

        # Has parameters for both args and kwargs
        if function_call.args and function_call.kwargs:
            return False

        return True

    def _find_library_import(self, library_name) -> Union[ImportStatement, ImportFromStatement]:
        for statement in self.lines:
            

    def step_1_check(self) -> Tuple[bool, str]:
        """Step Goal: Import tensorflow with the alias 'tf' and numpy with the alias 'np'.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        with open(self.script_path, "r") as f:
            file_lines = f.read().split("\n")

        if "import tensorflow as tf" not in file_lines:
            return (
                False,
                "Missing or incorrect tensorflow import line, did you import it with the specified alias?",
            )

        if "import numpy as np" not in file_lines:
            return (
                False,
                "Missing or incorrect numpy import line, did you import it with the specified alias?",
            )

        return True, ""

    def step_2_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Define dataset for OR operation.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = "np.array"
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name} isn't called"

        if len(function_calls) != 2:
            return (
                False,
                f"{function_name} should only be called twice, once for X and once for y",
            )

        func_call_for_X = FunctionCall.from_dict(function_calls[0])
        func_call_for_y = FunctionCall.from_dict(function_calls[1])

        if func_call_for_X.variable is None or func_call_for_y.variable is None:
            return False, f"{function_name} should be assigned to a variable"

        self.X = func_call_for_X.variable
        self.y = func_call_for_y.variable

        if func_call_for_X.args != ["[[0, 0], [0, 1], [1, 0], [1, 1]]"]:
            return False, f"Data passed to {self.X} doesn't match instructions."

        if func_call_for_y.args != ["[[0], [1], [1], [1]]"]:
            return False, f"Data passed to {self.y} doesn't match instructions."

        return True, ""

    def step_3_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Create single-layer perceptron with one dense layer, input shape of 2, sigmoid activation.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = "tf.keras.Sequential"
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name} isn't called"
        if len(function_calls) > 1:
            return False, f"{function_name} should only be called once"

        function_call = FunctionCall.from_dict(function_calls[0])
        if output_not_assigned_to_variable(function_call):
            return False, f"{function_name} output should be assigned to a variable"

        self.model = function_call.variable

        # Make sure that args only contains Dense layer with correct arguments
        solution_args = [
            "[tf.keras.layers.Input(shape=(2,)), tf.keras.layers.Dense(units=1, activation='sigmoid')]"
        ]
        if function_call.args != solution_args:
            return False, "Missing or incorrect layers for perceptron"

        return True, ""

    def step_4_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Compile the model with adam optimzier and binary_crossentropy loss function.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = f"{self.model}.compile"
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name}() isn't called"
        if len(function_calls) > 1:
            return False, f"{function_name}() should only be called once"

        function_call = FunctionCall.from_dict(function_calls[0])
        if function_call.variable is not None:
            return False, f"{function_name}() shouldn't be assigned to a variable"

        solution_kwargs = {"optimizer": "'adam'", "loss": "'binary_crossentropy'"}
        if function_call.kwargs != solution_kwargs:
            return (
                False,
                f"Missing or incorrect optimizer and loss function passed to {function_name}()",
            )

        return True, ""

    def step_5_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Train the model for 100 epochs.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = f"{self.model}.fit"
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
            return (
                False,
                f"Missing or incorrect parameters, are you passing your dataset and labels to {function_name}()?",
            )

        solution_kwargs = {"epochs": "100", "verbose": "1"}
        if function_call.kwargs != solution_kwargs:
            return (
                False,
                f"Missing or incorrect parameters, are you training for 100 epochs and outputting the training output?",
            )

        return True, ""

    def step_6_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Use the trained model to make predictions for X, rounding predictions to 0 or 1.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        # Validation for model.predict()
        predict_function_name = f"{self.model}.predict"
        function_calls = find_function_call(self.lines, predict_function_name)

        if function_not_called(function_calls):
            return False, f"{predict_function_name}() isn't called"
        if len(function_calls) > 1:
            return False, f"{predict_function_name}() should only be called once"

        function_call = FunctionCall.from_dict(function_calls[0])
        if output_not_assigned_to_variable(function_call):
            return (
                False,
                f"{predict_function_name}() output should be assigned to a variable",
            )

        self.predictions = function_call.variable

        solution_args = [self.X]
        if function_call.args != solution_args:
            return (
                False,
                f"{predict_function_name}() should take a single argument {self.X}",
            )

        solution_kwargs = {}
        if function_call.kwargs != solution_kwargs:
            return (
                False,
                f"{predict_function_name}() should not have any keyword arguments for this challenge",
            )

        return True, ""

    def step_7_check(self) -> Tuple[bool, str]:
        # Validation for predictions.round()
        function_name = f"{self.predictions}.round"
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name}() isn't called"
        if len(function_calls) > 1:
            return False, f"{function_name}() should only be called once"

        function_call = FunctionCall.from_dict(function_calls[0])

        if function_call.args != [] or function_call.kwargs != {}:
            return (
                False,
                f"{function_name}() shouldn't have any parameters passed to it for this challenge",
            )

        # Reassign self.predictions if user stored result in new variable after rounding
        if (
            function_call.variable is not None
            and function_call.variable != self.predictions
        ):
            self.predictions = function_call.variable

        return True, ""

    def step_8_check(self) -> Tuple[bool, str]:
        """
        Step Goal: Print the rounded predictions from the trained model.

        :return: A tuple containing a boolean indicating success or failure of the validation,
                and a string message providing error details if failure.
        :rtype: tuple[bool, str]
        """
        function_name = "print"
        function_calls = find_function_call(self.lines, function_name)

        if function_not_called(function_calls):
            return False, f"{function_name}() isn't called"
        if len(function_calls) > 1:
            return False, f"{function_name}() should only be called once"

        function_call = FunctionCall.from_dict(function_calls[0])

        solution_args = [self.predictions]
        if function_call.args != solution_args:
            return (
                False,
                f"Incorrect parameters passed to {function_name}(), are you passing your rounded predictions to the {function_name}() function?",
            )

        return True, ""

    def verify_code(self) -> None:
        """Runs the list of verification functions to make sure the user-submitted code
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

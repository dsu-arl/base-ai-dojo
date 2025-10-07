#!/usr/bin/exec-suid -- /usr/bin/python3.12 -I
import sys

sys.path.append("/challenge")

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from paceAITester.datatypes import AssignStatement, FunctionCallStatement, Statement
from paceAITester.parser import StatementParser
from paceAITester.utils import (
    find_function_calls,
    run_verification,
)


@dataclass
class UserVariables:
    """Stores the names of user variables"""

    x_train: Optional[Any] = None
    x_test: Optional[Any] = None
    y_train: Optional[Any] = None
    y_test: Optional[Any] = None
    model: Optional[Any] = None
    y_pred: Optional[Any] = None
    accuracy: Optional[Any] = None


class Validator:
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

    def step_1_check(self) -> Tuple[bool, str]:
        """Step Goal: Split the dataset where the test dataset is 20% of the original
        dataset with a random state of 42.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = "train_test_split"
        error_msg, statements = self._validate_basic_call(
            function_name, should_assign=True
        )
        if error_msg:
            return False, error_msg

        statement: AssignStatement = statements[0]
        function_call = self._get_function_call(statement)

        if len(statement.targets) != 4:
            return False, f"{function_name}() should be unpacked into 4 variables"

        if function_call.args != ["X", "y"]:
            return False, (
                f"You need to pass X and y to {function_name}() "
                "so it knows what data to split"
            )

        solution_kwargs = {"test_size": "0.2", "random_state": "42"}
        if function_call.kwargs != solution_kwargs:
            return False, (
                f"{function_name}() parameters don't match what's "
                "expected, did you give 'test_size' and 'random_state' the "
                "correct values from the instructions?"
            )

        self.user_vars.x_train = statement.targets[0]
        self.user_vars.x_test = statement.targets[1]
        self.user_vars.y_train = statement.targets[2]
        self.user_vars.y_test = statement.targets[3]

        return True, ""

    def step_2_check(self) -> Tuple[bool, str]:
        """Step Goal: Initialize a decision tree classifier model.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = "DecisionTreeClassifier"
        error_msg, statements = self._validate_basic_call(
            function_name, should_assign=True
        )
        if error_msg:
            return False, error_msg

        statement: AssignStatement = statements[0]
        function_call = self._get_function_call(statement)

        if function_call.args != [] or function_call.kwargs != {}:
            return False, (
                f"You shouldn't be passing any parameters to {function_name}() "
                "for this challenge"
            )

        self.user_vars.model = statement.targets[0]
        return True, ""

    def step_3_check(self) -> Tuple[bool, str]:
        """Step Goal: Fit the decision tree model to the training dataset.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = f"{self.user_vars.model}.fit"
        error_msg, statements = self._validate_basic_call(
            function_name, should_assign=False
        )
        if error_msg:
            return False, error_msg

        function_call = self._get_function_call(statements[0], allow_assign=False)

        solution_args = solution_args = [self.user_vars.x_train, self.user_vars.y_train]
        solution_kwargs = {"X": self.user_vars.x_train, "y": self.user_vars.y_train}

        if not self._validate_args(function_call, solution_args, solution_kwargs):
            return False, (
                f"Incorrect parameters for {function_name}(), are you correctly "
                "passing the training data to it?"
            )

        return True, ""

    def step_4_check(self) -> Tuple[bool, str]:
        """Step Goal: Make predictions on the test dataset using the trained decision
        tree model.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = f"{self.user_vars.model}.predict"
        error_msg, statements = self._validate_basic_call(
            function_name, should_assign=True
        )
        if error_msg:
            return False, error_msg

        statement: AssignStatement = statements[0]
        function_call = self._get_function_call(statement)

        solution_args = [self.user_vars.x_test]
        solution_kwargs = {"X": self.user_vars.x_test}

        if not self._validate_args(function_call, solution_args, solution_kwargs):
            return False, (
                f"Incorrect parameters for {function_name}(), are you correctly "
                "passing the test data to it?"
            )

        self.user_vars.y_pred = statement.targets[0]
        return True, ""

    def step_5_check(self) -> Tuple[bool, str]:
        """Step Goal: Retrieve the model's test dataset accuracy.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = "accuracy_score"
        error_msg, statements = self._validate_basic_call(
            function_name, should_assign=True
        )
        if error_msg:
            return False, error_msg

        statement: AssignStatement = statements[0]
        function_call = self._get_function_call(statement)

        solution_args = [self.user_vars.y_test, self.user_vars.y_pred]
        solution_kwargs = {
            "y_true": self.user_vars.y_test,
            "y_pred": self.user_vars.y_pred,
        }

        if not self._validate_args(function_call, solution_args, solution_kwargs):
            return False, (
                f"Incorrect parameters for {function_name}(), are you correctly "
                "passing the test output and model output to it?"
            )

        self.user_vars.accuracy = statement.targets[0]
        return True, ""

    def step_6_check(self) -> Tuple[bool, str]:
        """Step Goal: Print out the model's accuracy.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = "print"
        error_msg, statements = self._validate_basic_call(
            function_name, should_assign=False
        )
        if error_msg:
            return False, error_msg

        function_call = self._get_function_call(statements[0], allow_assign=False)

        if self.user_vars.accuracy not in function_call.args:
            return False, "Are you printing out the correct variable for the accuracy?"

        if function_call.kwargs != {} and function_call.args:
            return (
                False,
                "You don't need any keyword arguments for this print statement",
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

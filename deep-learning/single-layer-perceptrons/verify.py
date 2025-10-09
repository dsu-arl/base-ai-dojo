#!/usr/bin/exec-suid -- /usr/bin/python3.12 -I
import sys

sys.path.append("/challenge")

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from paceAITester.datatypes import (
    AssignStatement,
    FunctionCallStatement,
    ImportStatement,
    ImportFromStatement,
    Statement,
)
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

    def _find_library_import(
        self, library_name
    ) -> Optional[Union[ImportStatement, ImportFromStatement]]:
        for statement in self.lines:
            if isinstance(statement, ImportStatement) and statement.names == [
                library_name
            ]:
                return statement
            if (
                isinstance(statement, ImportFromStatement)
                and statement.module == library_name
            ):
                return statement
        return None

    def step_1_check(self) -> Tuple[bool, str]:
        """Step Goal: Import tensorflow with the alias 'tf' and numpy with the alias
        'np'.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        tensorflow_import = self._find_library_import("tensorflow")
        if tensorflow_import is None:
            return False, "Missing or incorrect tensorflow import"

        if tensorflow_import.alias != "tf":
            return False, "Missing or incorrect tensorflow import alias"

        numpy_import = self._find_library_import("numpy")
        if numpy_import is None:
            return False, "Missing or incorrect numpy import"

        if numpy_import.alias != "np":
            return False, "Missing or incorrect numpy import alias"

        return True, ""

    def step_2_check(self) -> Tuple[bool, str]:
        """Step Goal: Define dataset for OR operation.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = "np.array"
        statements = find_function_calls(self.lines, function_name)

        if len(statements) == 0:
            return False, "np.array isn't called"

        if len(statements) != 2:
            return (
                False,
                "np.array should only be called twice, once for X and once for y",
            )

        # Make sure all statements are assignments
        if not all(isinstance(statement, AssignStatement) for statement in statements):
            return False, "np.array should be assigned to X and y variables."

        correct_input_statement = FunctionCallStatement(
            func="np.array", args=["[[0, 0], [0, 1], [1, 0], [1, 1]]"], kwargs={}
        )
        correct_output_statement = FunctionCallStatement(
            func="np.array", args=["[[0], [1], [1], [1]]"], kwargs={}
        )
        input_data_statement = None
        output_data_statement = None
        for statement in statements:
            if (
                statement.value == correct_input_statement
                and input_data_statement is None
            ):
                input_data_statement = statement
            elif (
                statement.value == correct_output_statement
                and output_data_statement is None
            ):
                output_data_statement = statement

        if input_data_statement is None:
            return False, "Input data passed to np.array doesn't match instructions."

        if output_data_statement is None:
            return False, "Output data passed to np.array doesn't match instructions."

        self.user_vars.x = input_data_statement.targets[0]
        self.user_vars.y = output_data_statement.targets[0]

        return True, ""

    def step_3_check(self) -> Tuple[bool, str]:
        """Step Goal: Create single-layer perceptron with one dense layer, input shape
        of 2, sigmoid activation.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = "tf.keras.Sequential"
        error_msg, statements = self._validate_basic_call(
            function_name, should_assign=True
        )
        if error_msg:
            return False, error_msg

        statement: AssignStatement = statements[0]
        function_call = self._get_function_call(statement)

        correct_function_call = FunctionCallStatement(
            func="tf.keras.Sequential",
            args=[
                "[tf.keras.layers.Input(shape=(2,)), "
                "tf.keras.layers.Dense(units=1, activation='sigmoid')]"
            ],
            kwargs={},
        )
        if function_call != correct_function_call:
            return False, "Missing or incorrect layers for perceptron"

        self.user_vars.model = statement.targets[0]

        return True, ""

    def step_4_check(self) -> Tuple[bool, str]:
        """Step Goal: Compile the model with adam optimzier and binary_crossentropy loss
        function.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = f"{self.user_vars.model}.compile"
        error_msg, statements = self._validate_basic_call(
            function_name, should_assign=False
        )
        if error_msg:
            return False, error_msg

        function_call: FunctionCallStatement = statements[0]

        solution_kwargs = {"optimizer": "'adam'", "loss": "'binary_crossentropy'"}
        if function_call.kwargs != solution_kwargs:
            return (
                False,
                "Missing or incorrect optimizer and loss "
                "function passed to model.compile()",
            )

        return True, ""

    def step_5_check(self) -> Tuple[bool, str]:
        """Step Goal: Train the model for 100 epochs.

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

        function_call: FunctionCallStatement = statements[0]

        solution_args = [self.user_vars.x, self.user_vars.y]
        if function_call.args != solution_args:
            return (
                False,
                "Missing or incorrect parameters, are you passing your "
                "dataset and labels to model.fit()?",
            )

        solution_kwargs = {"epochs": "100", "verbose": "1"}
        if function_call.kwargs != solution_kwargs:
            return (
                False,
                "Missing or incorrect parameters, are you training for "
                "100 epochs and outputting the training output?",
            )

        return True, ""

    def step_6_check(self) -> Tuple[bool, str]:
        """Step Goal: Use the trained model to make predictions for X.

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

        if function_call.args != [self.user_vars.x]:
            return False, "model.predict() should take a single argument X"

        if function_call.kwargs != {}:
            return (
                False,
                "model.predict() should not have any keyword "
                "arguments for this challenge",
            )

        self.user_vars.predictions = statement.targets[0]

        return True, ""

    def step_7_check(self) -> Tuple[bool, str]:
        """Step Goal: Round the predicted value to either 0 or 1.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = f"{self.user_vars.predictions}.round"
        error_msg, statements = self._validate_basic_call(
            function_name, should_assign=True
        )
        if error_msg:
            return False, error_msg

        statement: AssignStatement = statements[0]
        function_call = self._get_function_call(statement)

        if function_call.args != [] or function_call.kwargs != {}:
            return (
                False,
                f"{function_name}() shouldn't have any parameters "
                "passed to it for this challenge",
            )

        self.user_vars.predictions = statement.targets[0]

        return True, ""

    def step_8_check(self) -> Tuple[bool, str]:
        """Step Goal: Print the rounded predictions from the trained model.

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

        function_call: FunctionCallStatement = statements[0]

        solution_args = [self.user_vars.predictions]
        if function_call.args != solution_args:
            return (
                False,
                f"Incorrect parameters passed to {function_name}(), are you passing "
                f"your rounded predictions to the {function_name}() function?",
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

#!/usr/bin/exec-suid -- /usr/bin/python3.12 -I
import sys

sys.path.append("/challenge")

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from paceAITester.datatypes import (
    AssignStatement,
    ClassDefStatement,
    FunctionCallStatement,
    FunctionDefStatement,
    GenericStatement,
    ImportStatement,
    ImportFromStatement,
    Statement,
)
from paceAITester.parser import StatementParser
from paceAITester.utils import (
    find_class_definition,
    find_function_calls,
    run_verification,
)


@dataclass
class DataUserVariables:
    """Stores the names of user variables for data"""

    x: Optional[Any] = None
    y: Optional[Any] = None
    outputs: Optional[Any] = None
    loss: Optional[Any] = None
    predictions: Optional[Any] = None


@dataclass
class ClassUserVariables:
    """Stores the names of user variables in the Perceptron class"""

    linear: Optional[Any] = None
    sigmoid: Optional[Any] = None
    x: Optional[Any] = None


@dataclass
class ModelUserVariables:
    """Stores the names of user variables for the perceptron model"""

    model: Optional[Any] = None
    criterion: Optional[Any] = None
    optimizer: Optional[Any] = None


class Validator:
    """Validator class that runs verification checks on user submitted code for
    challenge.
    """

    def __init__(self, script_path: str):
        self.parser = StatementParser(script_path)
        self.lines = self.parser.parse()
        self.data_user_vars = DataUserVariables()
        self.class_user_vars = ClassUserVariables()
        self.model_user_vars = ModelUserVariables()
        self.checks = [
            self.step_1_check,
            self.step_2_check,
            self.step_3_check,
            self.step_4_check,
            self.step_5_check,
            self.step_6_check,
            self.step_7_check,
            self.step_8_check,
            self.step_9_check,
        ]

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

    def _find_function_definition(
        self, function_name: str, statements: List[Statement] = None
    ) -> Optional[FunctionDefStatement]:
        if statements is None:
            statements = self.lines

        for statement in statements:
            if (
                isinstance(statement, FunctionDefStatement)
                and statement.name == function_name
            ):
                return statement

        return None

    def _find_function_call(
        self, function_name: str, statements: List[Statement] = None
    ) -> Optional[FunctionCallStatement]:
        if statements is None:
            statements = self.lines

        for statement in statements:
            if (
                isinstance(statement, FunctionCallStatement)
                and statement.func == function_name
            ):
                return statement

        return None

    def _find_assignment(
        self,
        value: Union[FunctionCallStatement, str],
        statements: List[Statement] = None,
    ) -> Optional[AssignStatement]:
        if statements is None:
            statements = self.lines

        for statement in statements:
            if isinstance(statement, AssignStatement) and statement.value == value:
                return statement

        return None

    def _validate_init_method(
        self, perceptron_class: ClassDefStatement
    ) -> Tuple[bool, str]:
        init_method = self._find_function_definition(
            "__init__", statements=perceptron_class.body
        )
        if init_method is None:
            return False, "Missing __init__ method in Perceptron class"

        # Incorrect init method conditions
        init_has_extra_params = init_method.args != ["self"]
        missing_super_call = (
            self._find_function_call("super().__init__", statements=init_method.body)
            is None
        )
        linear_function_call = self._find_assignment(
            FunctionCallStatement(func="nn.Linear", args=["2", "1"], kwargs={}),
            statements=init_method.body,
        )
        sigmoid_function_call = self._find_assignment(
            FunctionCallStatement(func="nn.Sigmoid", args=[], kwargs={}),
            statements=init_method.body,
        )

        incorrect_init_method = any(
            [
                init_has_extra_params,
                missing_super_call,
                linear_function_call is None,
                sigmoid_function_call is None,
            ]
        )
        if incorrect_init_method:
            return False, "Incorrectly defined __init__ method in Perceptron class"

        self.class_user_vars.linear = linear_function_call.targets[0]
        self.class_user_vars.sigmoid = sigmoid_function_call.targets[0]

        return True, ""

    def _validate_forward_method(
        self, perceptron_class: ClassDefStatement
    ) -> Tuple[bool, str]:
        forward_method = self._find_function_definition(
            "forward", statements=perceptron_class.body
        )
        if forward_method is None:
            return False, "Missing forward method in Perceptron class"

        # Incorrect forward method conditions
        forward_has_extra_params = len(forward_method.args) != 2
        if forward_has_extra_params:
            return False, "Incorrectly defined forward method in Perceptron class"

        self.class_user_vars.x = forward_method.args[1]

        linear_function_call: AssignStatement = self._find_assignment(
            FunctionCallStatement(
                func=self.class_user_vars.linear,
                args=[self.class_user_vars.x],
                kwargs={},
            ),
            statements=forward_method.body,
        )
        if linear_function_call is None:
            return False, "Incorrectly defined forward method in Perceptron class"

        # Update x variable name in case user used a different variable name
        self.class_user_vars.x = linear_function_call.targets[0]
        sigmoid_function_call = self._find_assignment(
            FunctionCallStatement(
                func=self.class_user_vars.sigmoid,
                args=[self.class_user_vars.x],
                kwargs={},
            ),
            statements=forward_method.body,
        )

        if sigmoid_function_call is None:
            return False, "Incorrectly defined forward method in Perceptron class"

        if forward_method.body[-1] != GenericStatement(type_name="Return"):
            return False, "Incorrectly defined forward method in Perceptron class"

        return True, ""

    def _validate_nparray_data(self, nparray_type: str) -> Tuple[bool, str]:
        if nparray_type == "input":
            nparray_data = ["[[0, 0], [0, 1], [1, 0], [1, 1]]"]
        elif nparray_type == "output":
            nparray_data = ["[[0], [0], [0], [1]]"]
        else:
            raise ValueError(
                f"nparray_type must be either 'input' or 'output', got '{nparray_type}'"
            )

        data_statement: AssignStatement = self._find_assignment(
            FunctionCallStatement(
                func="np.array", args=nparray_data, kwargs={"dtype": "np.float32"}
            )
        )

        if data_statement is None:
            return (
                False,
                f"Missing or incorrect np.array statement for {nparray_type} data",
            )

        if nparray_type == "input":
            self.data_user_vars.x = data_statement.targets[0]
        else:
            self.data_user_vars.y = data_statement.targets[0]

        return True, ""

    def _validate_tensor_convert(self, nparray_type: str) -> Tuple[bool, str]:
        if nparray_type == "input":
            array_name = self.data_user_vars.x
        elif nparray_type == "output":
            array_name = self.data_user_vars.y
        else:
            raise ValueError(
                f"nparray_type must be either 'input' or 'output', got '{nparray_type}'"
            )

        statement: AssignStatement = self._find_assignment(
            FunctionCallStatement(func="torch.from_numpy", args=[array_name], kwargs={})
        )
        if statement is None:
            return False, f"Missing {nparray_type} data conversion to PyTorch tensor"

        if nparray_type == "input":
            self.data_user_vars.x = statement.targets[0]
        else:
            self.data_user_vars.y = statement.targets[0]

        return True, ""

    def step_1_check(self) -> Tuple[bool, str]:
        """Step Goal: Import the libraries torch, torch.nn with alias 'nn', and numpy
        with alias 'np'.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        torch_import = self._find_library_import("torch")
        if torch_import is None:
            return False, "Missing or incorrect torch import"

        torchnn_import = self._find_library_import("torch.nn")
        if torchnn_import is None:
            return False, "Missing or incorrect torch.nn import"
        if torchnn_import.alias != "nn":
            return False, "Missing or incorrect torch.nn import alias"

        numpy_import = self._find_library_import("numpy")
        if numpy_import is None:
            return False, "Missing or incorrect numpy import"
        if numpy_import.alias != "np":
            return False, "Missing or incorrect numpy import alias"

        return True, ""

    def step_2_check(self) -> Tuple[bool, str]:
        """Step Goal: Build a single-layer perceptron class called Perceptron for model
        using 'nn.Module'.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        perceptron_class = find_class_definition(self.lines, "Perceptron")
        if perceptron_class is None or perceptron_class.bases != ["nn.Module"]:
            return False, "Missing or incorrectly defined Perceptron class"

        is_valid, error_msg = self._validate_init_method(perceptron_class)
        if not is_valid:
            return False, error_msg

        is_valid, error_msg = self._validate_forward_method(perceptron_class)
        if not is_valid:
            return False, error_msg

        return True, ""

    def step_3_check(self) -> Tuple[bool, str]:
        """Step Goal: Define dataset for AND operation.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = "np.array"
        statements = find_function_calls(self.lines, function_name)

        if len(statements) != 2:
            return (
                False,
                "np.array should only be called twice, once for X and once for y",
            )

        for data_type in ["input", "output"]:
            is_valid, error_msg = self._validate_nparray_data(data_type)
            if not is_valid:
                return False, error_msg

        return True, ""

    def step_4_check(self) -> Tuple[bool, str]:
        """Step Goal: Convert NumPy arrays to PyTorch tensors.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        for data_type in ["input", "output"]:
            is_valid, error_msg = self._validate_tensor_convert(data_type)
            if not is_valid:
                return False, error_msg

        return True, ""

    def step_5_check(self) -> Tuple[bool, str]:
        # Checks if perceptron model is correctly instantiated
        pass

    def step_6_check(self) -> Tuple[bool, str]:
        # Checks if loss function and optimizer are correctly defined
        pass

    def step_7_check(self) -> Tuple[bool, str]:
        # Checks if training loop is correctly implemented
        # Forward pass, clear previous gradients, compute gradients, update weights
        pass

    def step_8_check(self) -> Tuple[bool, str]:
        # Make predictions using trained model and round them
        pass

    def step_9_check(self) -> Tuple[bool, str]:
        # Checks if rounded predictions are converted back to NumPy array and printed out
        pass

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

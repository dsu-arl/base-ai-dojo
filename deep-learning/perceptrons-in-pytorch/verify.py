#!/usr/bin/exec-suid -- /usr/bin/python3.12 -I
import sys

sys.path.append("/challenge")

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
from paceAITester.datatypes import (
    AssignStatement,
    ClassDefStatement,
    ForStatement,
    FunctionCallStatement,
    FunctionDefStatement,
    GenericStatement,
    ImportStatement,
    ImportFromStatement,
    Statement,
    WithStatement,
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
    ) -> Optional[AssignStatement | FunctionCallStatement]:
        if statements is None:
            statements = self.lines

        for statement in statements:
            if (
                isinstance(statement, FunctionCallStatement)
                and statement.func == function_name
            ) or (
                isinstance(statement, AssignStatement)
                and statement.value.func == function_name
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

    def _validate_loss_function(self) -> Tuple[bool, str]:
        function_name = "nn.BCELoss"
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
                f"You shouldn't be passing any parameters to {function_name}() "
                "for this challenge",
            )

        self.model_user_vars.criterion = statement.targets[0]
        return True, ""

    def _validate_optimizer(self) -> Tuple[bool, str]:
        function_name = "torch.optim.Adam"
        error_msg, statements = self._validate_basic_call(
            function_name, should_assign=True
        )
        if error_msg:
            return False, error_msg

        statement: AssignStatement = statements[0]
        function_call = self._get_function_call(statement)

        correct_args = [f"{self.model_user_vars.model}.parameters()"]
        if function_call.args != correct_args or function_call.kwargs != {}:
            return (
                False,
                f"Incorrect parameters for {function_name}(), are you correctly "
                "passing the model parameters to it?",
            )

        self.model_user_vars.optimizer = statement.targets[0]
        return True, ""

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

    def _validate_epochs(self, for_loop_iter: str) -> Tuple[bool, str]:
        if for_loop_iter.startswith("range(") and for_loop_iter.endswith(")"):
            epochs_variable_name = for_loop_iter.split("(")[1].split(")")[0]
            if epochs_variable_name.isdigit():
                if int(epochs_variable_name) != 100:
                    return False, "Incorrect epoch count for training loop"
            else:
                variables = self.parser.retrieve_variable_values()
                if (
                    epochs_variable_name in variables
                    and variables[epochs_variable_name] != 100
                ):
                    return False, "Incorrect epoch count for training loop"
        else:
            return False, "Missing training loop"

        return True, ""

    def _validate_forward_pass(
        self, for_loop_body: List[Statement]
    ) -> Tuple[bool, str]:
        model_predictions = self._find_assignment(
            value=FunctionCallStatement(
                func=self.model_user_vars.model, args=[self.data_user_vars.x], kwargs={}
            ),
            statements=for_loop_body,
        )
        if model_predictions is None:
            return False, "Missing or incorrect forward pass in training loop"

        self.data_user_vars.outputs = model_predictions.targets[0]

        calculate_loss = self._find_assignment(
            value=FunctionCallStatement(
                func=self.model_user_vars.criterion,
                args=[self.data_user_vars.outputs, self.data_user_vars.y],
                kwargs={},
            ),
            statements=for_loop_body,
        )
        if calculate_loss is None:
            return False, "Missing or incorrect forward pass in training loop"

        self.data_user_vars.loss = calculate_loss.targets[0]
        return True, ""

    def _validate_backward_pass(
        self, for_loop_body: List[Statement]
    ) -> Tuple[bool, str]:
        clear_gradients = self._find_function_call(
            f"{self.model_user_vars.optimizer}.zero_grad", for_loop_body
        )
        compute_gradients = self._find_function_call(
            f"{self.data_user_vars.loss}.backward", for_loop_body
        )
        update_weights = self._find_function_call(
            f"{self.model_user_vars.optimizer}.step", for_loop_body
        )
        if (
            clear_gradients is None
            or compute_gradients is None
            or update_weights is None
        ):
            return False, "Missing or incorrect backward pass in training loop"

        return True, ""

    def _validate_inference_call(self, with_body: List[Statement]) -> Tuple[bool, str]:
        inference_statement = self._find_function_call(
            function_name=self.model_user_vars.model, statements=with_body
        )
        if inference_statement is None:
            return False, "Missing code to make predictions using the trained model"
        if isinstance(inference_statement, FunctionCallStatement):
            return False, "Prediction output not stored in a variable"
        if (
            inference_statement.value.args != [self.data_user_vars.x]
            or inference_statement.value.kwargs != {}
        ):
            return False, "Incorrect data passed to model for inference"

        self.data_user_vars.predictions = inference_statement.targets[0]
        return True, ""

    def _validate_round_call(self, with_body: List[Statement]) -> Tuple[bool, str]:
        round_function_name = f"{self.data_user_vars.predictions}.round"
        round_statement = self._find_function_call(
            function_name=round_function_name, statements=with_body
        )
        if round_statement is None:
            return False, f"Missing {round_function_name}() call to round predictions"
        if isinstance(round_statement, FunctionCallStatement):
            return False, f"{round_function_name}() output not stored in a variable"
        if round_statement.value.args != [] or round_statement.value.kwargs != {}:
            return (
                False,
                f"You shouldn't be passing any parameters to {round_function_name}() "
                "for this challenge",
            )

        self.data_user_vars.predictions = round_statement.targets[0]
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
        """Step Goal: Instantiate the perceptron model.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        function_name = "Perceptron"
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
                f"You shouldn't be passing any parameters to {function_name}() "
                "for this challenge",
            )

        self.model_user_vars.model = statement.targets[0]

        return True, ""

    def step_6_check(self) -> Tuple[bool, str]:
        """Step Goal: Define the loss function and optimizer

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        for validate_function in [
            self._validate_loss_function,
            self._validate_optimizer,
        ]:
            is_valid, error_msg = validate_function()
            if not is_valid:
                return False, error_msg

        return True, ""

    def step_7_check(self) -> Tuple[bool, str]:
        """Step Goal: Define training loop.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        for_loop = None
        for statement in self.lines:
            if isinstance(statement, ForStatement):
                for_loop = statement
                break
        if for_loop is None:
            return False, "Missing training loop"

        for validation_func in [
            self._validate_epochs(for_loop.iter),
            self._validate_forward_pass(for_loop.body),
            self._validate_backward_pass(for_loop.body),
        ]:
            is_valid, error_msg = validation_func
            if not is_valid:
                return False, error_msg

        # make sure training loop contents are in correct order
        correct_for_loop_body = [
            AssignStatement(
                targets=[self.data_user_vars.outputs],
                value=FunctionCallStatement(
                    func=self.model_user_vars.model,
                    args=[self.data_user_vars.x],
                    kwargs={},
                ),
            ),
            AssignStatement(
                targets=[self.data_user_vars.loss],
                value=FunctionCallStatement(
                    func=self.model_user_vars.criterion,
                    args=[self.data_user_vars.outputs, self.data_user_vars.y],
                    kwargs={},
                ),
            ),
            FunctionCallStatement(
                func=f"{self.model_user_vars.optimizer}.zero_grad", args=[], kwargs={}
            ),
            FunctionCallStatement(
                func=f"{self.data_user_vars.loss}.backward", args=[], kwargs={}
            ),
            FunctionCallStatement(
                func=f"{self.model_user_vars.optimizer}.step", args=[], kwargs={}
            ),
        ]
        if for_loop.body[:5] != correct_for_loop_body:
            return False, "Incorrect training loop"

        return True, ""

    def step_8_check(self) -> Tuple[bool, str]:
        """Step Goal: Make predictions using the trained model and round them.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        with_statement = None
        for statement in self.lines:
            if isinstance(statement, WithStatement) and statement.items == [
                ("torch.no_grad()", None)
            ]:
                with_statement = statement
                break
        if with_statement is None:
            return False, "Missing code to disable gradient computation for inference"

        for validate_function in [
            self._validate_inference_call,
            self._validate_round_call,
        ]:
            is_valid, error_msg = validate_function(with_statement.body)
            if not is_valid:
                return False, error_msg

        return True, ""

    def step_9_check(self) -> Tuple[bool, str]:
        """Step Goal: Printed rounded predictions converted back to NumPy array.

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

        correct_args = [f"{self.data_user_vars.predictions}.numpy()"]
        if function_call.args != correct_args:
            return (
                False,
                "Incorrect data to be printed out or not in the "
                "required format for the challenge",
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

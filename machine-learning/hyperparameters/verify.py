#!/usr/bin/exec-suid -- /usr/bin/python3.12 -I
import sys

sys.path.append("/challenge")

from typing import Dict, Optional, Tuple
from paceAITester.config import GREEN_TEXT_CODE, RED_TEXT_CODE, RESET_CODE
from paceAITester.datatypes import (
    AssignStatement,
    FunctionCallStatement,
    ImportFromStatement,
)
from paceAITester.parser import StatementParser
from paceAITester.utils import print_flag


class Validator:
    """Validator class that runs verification checks on user submitted code for
    challenge.
    """

    def __init__(self, script_path: str):
        self.parser = StatementParser(script_path)
        self.lines = self.parser.parse()
        self.variables = self.parser.retrieve_variable_values()

    def _find_variable(self, variable_name: str) -> Optional[Dict[str, str]]:
        for key, value in self.variables.items():
            if key == variable_name:
                return {"name": key, "value": value}

        return None

    def check_template(self) -> Tuple[bool, str]:
        """Step Goal: Ensure that the template code is correctly used in Python file.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        correct_template = [
            ImportFromStatement(module="sklearn.datasets", names=["load_iris"]),
            ImportFromStatement(
                module="sklearn.ensemble", names=["RandomForestClassifier"]
            ),
            ImportFromStatement(module="sklearn.metrics", names=["log_loss"]),
            ImportFromStatement(
                module="sklearn.model_selection", names=["train_test_split"]
            ),
            AssignStatement(
                targets=["iris"],
                value=FunctionCallStatement(func="load_iris", args=[], kwargs={}),
            ),
            AssignStatement(targets=["X"], value="iris.data"),
            AssignStatement(targets=["y"], value="iris.target"),
            AssignStatement(
                targets=["X_train", "X_test", "y_train", "y_test"],
                value=FunctionCallStatement(
                    func="train_test_split",
                    args=["X", "y"],
                    kwargs={"test_size": "0.2", "random_state": "42"},
                ),
            ),
            AssignStatement(
                targets=["model"],
                value=FunctionCallStatement(
                    func="RandomForestClassifier",
                    args=[],
                    kwargs={"n_estimators": "n_estimators", "random_state": "42"},
                ),
            ),
            FunctionCallStatement(
                func="model.fit", args=["X_train", "y_train"], kwargs={}
            ),
            AssignStatement(
                targets=["y_pred_proba"],
                value=FunctionCallStatement(
                    func="model.predict_proba", args=["X_test"], kwargs={}
                ),
            ),
            AssignStatement(
                targets=["loss"],
                value=FunctionCallStatement(
                    func="log_loss", args=["y_test", "y_pred_proba"], kwargs={}
                ),
            ),
            FunctionCallStatement(
                func="print", args=["'Target Loss: 0.0252'"], kwargs={}
            ),
            FunctionCallStatement(
                func="print", args=["f' Model Loss: {loss:.4f}'"], kwargs={}
            ),
        ]

        # Remove line for "n_estimators = "
        n_estimators_line_removed = False
        filtered_list = self.lines.copy()
        for i, item in enumerate(filtered_list):
            if isinstance(item, AssignStatement) and item.targets == ["n_estimators"]:
                filtered_list.pop(i)
                n_estimators_line_removed = True
                break

        if filtered_list != correct_template or not n_estimators_line_removed:
            return (
                False,
                "No need to modify the given code template, only modify the "
                "'n_estimators' value",
            )

        return True, ""

    def check_answer(self) -> Tuple[bool, str]:
        """Step Goal: Ensure that the n_estimators variable has the correct value.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success or
                failure of the validation, and a string message providing error details
                if failure.
        """
        variable_call = self._find_variable("n_estimators")
        if variable_call is None:
            return False, "Make sure you have a variable called 'n_estimators'"

        if variable_call["value"] == "Unresolvable dynamic value" or not isinstance(
            variable_call["value"], int
        ):
            return (
                False,
                "Make sure you're assigning an integer value to 'n_estimators'",
            )

        guess = variable_call["value"]
        if guess != 140:
            if guess > 140:
                direction = "Lower"
            else:
                direction = "Higher"
            return False, f"Incorrect (HINT: {direction} 'n_estimators' value)"

        return True, ""

    def verify_code(self) -> None:
        """Runs verification functions to make sure the user-submitted code
        is correct.
        """
        template_correct, error_msg = self.check_template()
        if not template_correct:
            print(f"{RED_TEXT_CODE}{error_msg}{RESET_CODE}")
            sys.exit(1)

        passed, error_msg = self.check_answer()
        if not passed:
            print(f"{RED_TEXT_CODE}{error_msg}{RESET_CODE}")
            sys.exit(1)

        print(f"{GREEN_TEXT_CODE}Correct{RESET_CODE}")
        print("Congratulations! You have passed this challenge! Here is your flag:")
        print_flag()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: verify <user_script>.py")
        sys.exit(1)

    filepath = sys.argv[1]

    validator = Validator(filepath)
    validator.verify_code()

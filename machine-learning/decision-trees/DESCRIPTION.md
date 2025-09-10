In this challenge, you'll learn how to use a decision tree to classify flowers based on the length and width of sepals and petals. Decision trees are a supervised learning algorithm used for both classification and regression tasks, where they split data into branches based on feature values to make predictions. Each internal node represents a decision based on a feature, each branch represents an outcome of that decision, and each leaf node represents a final prediction or outcome.

As with the machine learning models that we have covered, we first need to import it from `sklearn` and then define it.
```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
```

We can train and make predictions from our decision tree the same way as the previous models.

To get this flag, create a Python file, copy and paste the following starter code, and then complete the steps listed below:
```python
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Initializing the data
iris = load_iris()
X = iris.data
y = iris.target
```

1. Split the dataset where the test dataset is 20% of the original and set a random state of 42
2. Initialize a decision tree classifier model
3. Train the decision tree classifier model on the iris dataset
4. Make predictions on the test iris dataset
5. Retrieve the model's test dataset accuracy and store it in a variable
6. Print out the model's accuracy

You can test your code by running `/usr/bin/python3 <your_file>.py`. When you're ready to verify your solution, you can run `verify <your_file>.py`.
The last machine learning model we'll cover is random forests. You already know more about this model than you think since a random forest is just a collection of decision trees! Random forests work by building multiple decision trees during training and then merging their results to output a single value which helps improve its accuracy.

Like our previous models, we'll import it from `sklearn` and then define it.
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
```

To get this flag, create a Python file, copy and paste the following starter code, and then complete the steps listed below:
```python
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Initializing the data
iris = load_iris()
X = iris.data
y = iris.target
```

1. Split the dataset where the test dataset is 30% of the original and set a random state of 23
2. Initialize a random forest model
3. Train the random forest model on the iris dataset
4. Make predictions on the test iris dataset
5. Retrieve the model's test dataset accuracy and store it in a variable
6. Print out the model's accuracy

You can test your code by running `/usr/bin/python3 <your_file>.py`. When you're ready to verify your solution, you can run `/challenge/verify <your_file>.py`.
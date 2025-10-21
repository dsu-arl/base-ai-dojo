Before we can train our model and make predictions with it, we first need to split up our dataset into a dataset used for training (train dataset) and a dataset used to evaluate our model (test dataset).

Imagine you're studying for a test. The train dataset is like your practice questions—it helps you learn and prepare. The test dataset is like the actual exam—questions you've never seen before that check if you really understand the material.

If you only use the practice questions to judge your learning, you might just memorize answers without actually understanding. The test dataset ensures your model is smart enough to handle new situations, not just repeat what it saw during practice!

When your dataset is ready to be split into training and testing, it will generally be stored in variables such as `X` and `y`. `X` is the input for the model and `y` is the correct output for the model. The model will compare its output with the values in `y` to know how much to adjust itself so that it can output the correct answer.

It is conventional to make `X` uppercase to symbolize that it is a matrix and `y` lowercase to symbolize it as a vector (array).

For this challenge, we will use `numpy` to generate some random data for us to split into training and test datasets!

```python
import numpy as np
```

First, we will create `X`, our model inputs, using `numpy`'s `random.rand()` function. The first parameter is how many samples we want and the second is how many features we want. The following example has 100 samples and 3 features.
```python
X = np.random.rand(100, 3)
```

Now we will create `y`, the correct output for the model to reference, using `numpy`'s `random.randint()` function. For this example we want our model to output either a no or yes (0 or 1). The first parameter is the low value, which in our case is 0. The second parameter is the high value and we'll set to be 2 since it's exclusive. This will only output 0s and 1s. The final parameter we will set is the size, which is the number of samples which is the same as the value set in the last example.
```python
y = np.random.randint(0, 2, size=100)
```

Now that we have our model inputs and outputs, we can finally split up the dataset into our train and test datasets!
To do this we will use the `train_test_split()` function from `sklearn.model_selection`.
```python
from sklearn.model_selection import train_test_split
```

There will be 3 parameters that we pass to `train_test_split()`:
1. Our inputs `X`
2. The correct outputs `y`
3. What percentage of the dataset the size of the test dataset should be (uses decimals so a value of 0.2 means 20%)

`train_test_split()` will output 4 values:
1. Inputs for our train dataset
2. Inputs for our test dataset
3. Correct outputs for our train dataset
4. Correct outputs for our test dataset

So putting all of that together we get the following:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

NOTE: Another useful parameter to set in `train_test_split()` is the `random_state` parameter which controls the randomness of the data splitting process, ensuring that you can reproduce the same split every time you run the code.

Create a Python file and copy and paste the following imports statements into the top of the file before starting on the steps:
```python
import numpy as np
from sklearn.model_selection import train_test_split
```

Complete the following steps to pass this challenge:
1. Create `X` using `np.random.rand()` which has 500 samples and 7 features
2. Create `y` using `np.random.randint()` which has 500 samples and outputs either 0 or 1
3. Split the dataset where the train dataset is 70% of the original dataset

You can test your code by running `/usr/bin/python3 <your_file>.py`. When you're ready to verify your solution, you can run `/challenge/verify <your_file>.py`.
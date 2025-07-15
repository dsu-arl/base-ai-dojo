### Single-Layer Perceptrons
A single-layer perceptron is a fundamental building block of neural networks, used for binary classification of linearly separable data. It consists of an input layer, weights, a bias, and an activation function to produce a binary output (0 or 1). While TensorFlow simplifies implementation, the perceptron's core concepts apply across all programming environments.

* **Input Layer:** Takes a vector of input features (e.g., a 2D input for two features).
* **Weights:** Each input is multiplied by a weight, determining its influence on the output.
* **Bias:** A constant added to the weighted sum, adjusting the decision boundary.
* **Activation Function:** Typically a step function (in classic perceptrons) or sigmoid (in modern implementations), mapping the weighted sum to a binary output.
* **Training Process:** Adjusts weights and bias to minimize prediction errors using a learning algorithm (e.g., gradient descent or the perceptron learning rule).
* **Limitations:** Can only solve linearly separable problems (e.g., AND, OR, but not XOR).
* **General Implementation:** Requires manual computation of weighted sums, activation, and weight updates using loops and basic math operations in languages like Python, C, or Java.

### Creating a Single-Layer Perceptron with TensorFlow

TensorFlow's Keras API simplifies perceptron implementation by abstracting low-level math. You define a model with a Dense layer, specify an optimizer and loss function, and train it on data. TensorFlow handles backpropagation and optimization, making it efficient for rapid prototyping and scalable applications.

#### Importing Libraries
TensorFlow provides tools for building and training neural networks, while NumPy is used for efficient array operations. Importing these libraries sets up the environment for creating and manipulating the perceptron’s data and model.
```python
import tensorflow as tf
import numpy as np
```

#### Creating Sample Data with NumPy
This step generates a small dataset to simulate the logical AND operation, where the output is 1 if at least one input is 1. The input array `X` contains four pairs of binary inputs, and the output array `y` provides the corresponding AND labels. NumPy arrays are used for compatibility with TensorFlow.
```python
# Sample data ('AND' operation)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])
```

#### Defining the Perceptron
The perceptron is defined using Keras’ `Sequential` API, which allows stacking layers. A single `Dense` layer with one unit is used, taking a 2D input (for two features) and applying a sigmoid activation function to produce a probability-like output between 0 and 1, suitable for binary classification.
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[2], activation='sigmoid')
])
```

#### Compiling the Model
Compiling configures the model for training. The `adam` optimizer efficiently adjusts weights using adaptive learning rates. The `binary_crossentropy` loss function measures the error between predicted probabilities and true binary labels, ideal for binary classification tasks like AND.
```python
model.compile(optimizer='adam', loss='binary_crossentropy')
```

#### Training the Model
Training involves feeding the input data `X` and labels `y` to the model for 100 epochs (iterations over the dataset). The `verbose=0` argument suppresses training logs for cleaner output. During training, TensorFlow adjusts the perceptron’s weights and bias to minimize the loss.
```python
model.fit(X, y, epochs=100, verbose=0)
```

#### Making Predictions with the Trained Model
After training, the model predicts outputs for the input data `X`. The predictions are probabilities (due to the sigmoid activation), which can be rounded to 0 or 1 using `.round()` for binary classification. This step evaluates how well the perceptron learned the AND operation.
```python
predictions = model.predict(X)
predictions = predictions.round()
```

### Challenge Instructions
Create a new Python file in your home directory and follow these steps to complete this challenge!
1. Import the required libraries: `tensorflow as tf` and `numpy as np`.
2. Define a dataset with the following:
    * Input `X`: A 4x2 NumPy array `[[0, 0], [0, 1], [1, 0], [1, 1]]` representing input pairs.
    * Output `y`: A 4x1 NumPy array `[[0], [1], [1], [1]]` representing OR-like labels (1 when at least one input is 1).
3. Build a single-layer perceptron using `tf.keras.Sequential` with:
    * One `Dense` layer with 1 unit
    * Input shape of `2`
    * `sigmoid` activation to output values between 0 and 1
4. Compile the model with:
    * `adam` optimizer for efficient gradient descent.
    * `binary_crossentropy` loss function, suitable for binary classification.
5. Train the model on `X` and `y` for 1000 epochs with `verbose=0` to suppress training output.
6. Use the trained model to predict outputs for `X`, rounding predictions to 0 or 1 using `.round()`.
7. Round predictions from the model to 0 or 1 using `.round()`.
8. Print the rounded predictions.
9. Save your program and run it to test the output.
10. To get the flag, run `/challenge/verify <yourfile>.py` to verify your solution.

Example output:
```commandline
[[0.]
 [1.]
 [1.]
 [1.]]
```
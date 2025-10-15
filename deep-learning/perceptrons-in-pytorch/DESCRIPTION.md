Now that you know what a perceptron is and how to build one using TensorFlow, this challenge will cover building a perceptron using PyTorch.

### 1. Importing Libraries
```python
import torch
import torch.nn as nn
import numpy as np
```
Let’s start with the tools we need. Think of `torch` as the heart of PyTorch—it gives us tensors, which are like super-powered arrays that can run on GPUs and track gradients for learning. We need this for all our computations. Then, `torch.nn` (aliased as `nn`) is where we get the building blocks for neural networks, like layers and loss functions. We’ll use it to define our perceptron’s structure and how it measures errors. Finally, `numpy` is our go-to for creating the dataset. It’s great for handling arrays before we turn them into PyTorch tensors. These imports are like grabbing your toolbox before starting a project—you can’t build anything without them.

### 2. Defining the Perceptron Model
```python
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
```
Now we’re building the perceptron itself, and in PyTorch, we do this by creating a custom class. Think of this as designing a little machine. We make it inherit from `nn.Module` because that gives us PyTorch’s magic for handling layers and gradients. In the `__init__` method, we set up two parts: a `nn.Linear` layer, which is the core of our perceptron—it takes our inputs (2 features for OR), multiplies them by weights, adds a bias, and gives a single output. Then, we add a `nn.Sigmoid` to squash that output into a 0-to-1 range, perfect for binary classification since we want to predict 0 or 1.

The `forward` method is where the action happens: it says, “Hey, when you get some input `x`, run it through the linear layer, then through the sigmoid, and spit out the result.” This is how our perceptron processes data, and PyTorch uses this to figure out gradients when we train. It’s like telling the machine exactly how to transform inputs into predictions.

### 3. Creating Sample Data
```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [1]], dtype=np.float32)
```
Here’s where we set up our data for the OR problem. We’re creating a tiny dataset with four input pairs: `[0, 0]`, `[0, 1]`, `[1, 0]`, and `[1, 1]`. These represent all possible combinations of two binary inputs. The labels in `y` tell us what OR should output: 0 when both inputs are 0, and 1 otherwise. We use NumPy because it’s super easy to create these arrays, and we set `dtype=np.float32` to match PyTorch’s default precision for tensors. Think of this as setting up a little practice problem for our perceptron to learn from—it’s like giving it a worksheet with questions and answers.

### 4. Converting Data to PyTorch Tensors
```python
X = torch.from_numpy(X)
y = torch.from_numpy(y)
```
Our perceptron speaks PyTorch tensors, not NumPy arrays, so we need to convert our data. The `torch.from_numpy` function takes our NumPy arrays and turns them into tensors, which are ready for the model to chew on. Tensors are special because they can live on a GPU and track gradients, which we’ll need when training. This step is like translating our worksheet into a language the perceptron understands.

### 5. Instantiating the Model
```python
model = Perceptron()
```
Now we fire up our perceptron by creating an instance of the `Perceptron` class, telling it we have 2 input features (since our OR data has two inputs per sample). This sets up the model with random weights and a bias, ready to learn. It’s like turning on the machine we built, plugging it in, and getting it ready to start processing data.

### 6. Defining Loss Function and Optimizer
```python
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters())  # Adam optimizer
```
Next, we need to tell the perceptron how to measure its mistakes and improve. The `nn.BCELoss` is our loss function—it compares the model’s 0-to-1 predictions (from the sigmoid) to the true 0 or 1 labels and gives us a number representing how wrong we are. Binary cross-entropy is perfect for this because it’s designed for binary classification, like our OR problem.

The `optimizer` is where we pick Adam, a smart algorithm that adjusts the model’s weights to reduce the loss. We pass it `model.parameters()`, which are the weights and bias in our linear layer, so it knows what to tweak. Think of the loss function as a scorekeeper telling us how bad the perceptron’s guesses are, and the optimizer as the coach making adjustments to improve those guesses.

### 7. Training Loop
```python
epochs = 100
for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```
This is where the learning happens, and it’s the most hands-on part in PyTorch. We loop 100 times (`epochs = 100`) to give the perceptron enough chances to get good at OR. In each loop:
- We feed `X` through the model to get `outputs`—our predictions.
- We calculate the `loss` by comparing those predictions to the true labels `y` using our loss function.
- We call `optimizer.zero_grad()` to clear out old gradients (otherwise, they’d pile up and mess things up).
- `loss.backward()` tells PyTorch to figure out how much each weight contributed to the error, computing gradients.
- `optimizer.step()` updates the weights based on those gradients, nudging the model toward better predictions.
- Every 10 epochs, we print the loss to see how the model’s doing—it should get smaller as the perceptron learns.

This loop is like a training session: the perceptron makes guesses, checks its answers, learns from its mistakes, and keeps improving.

### 8. Making Predictions
```python
with torch.no_grad():
    predictions = model(X)
    predictions = predictions.round()
```
Once the model’s trained, we want to see what it’s learned. We use `torch.no_grad()` to turn off gradient tracking since we’re just predicting, not training—this saves memory and makes things faster. We pass `X` through the model to get predictions, which are probabilities between 0 and 1 because of the sigmoid. Then, we `.round()` them to get 0 or 1, matching the binary outputs we want for OR. It’s like asking the perceptron, “Alright, show me what you’ve got on the test data.”

9. Printing Predictions
```python
print(predictions.numpy())
```
Finally, we use `.numpy()` to convert the tensor back to a NumPy array like `[[0.], [1.], [1.], [1.]]`.

### Challenge Instructions
Create a new Python file in `/home/hacker` directory and follow these steps to complete this challenge!
1. Import the required libraries with these aliases:
    * `torch` (no alias)
    * `torch.nn as nn`
    * `numpy as np`.
2. Build a single-layer perceptron class called Perceptron for our model using `nn.Module` with:
    * Linear layer that accepts 2 inputs and outputs a single value
    * `sigmoid` activation function to output values between 0 and 1
3. Define a dataset with the following:
    * Input `X`: A 4x2 NumPy array `[[0, 0], [0, 1], [1, 0], [1, 1]]` representing input pairs.
    * Output `y`: A 4x1 NumPy array `[[0], [0], [0], [1]]` representing logical AND labels (1 only if both inputs are 1).
    * Set the datatype of both to be `np.float32`.
4. Convert the data from NumPy arrays to PyTorch tensors.
5. Initialize the perceptron model.
6. Create loss function and optimizer variables for the model:
    * `nn.BCELoss` loss function, suitable for binary classification.
    * `torch.optim.Adam` optimizer for efficient gradient descent.
7. Train the model on `X` and `y` for 100 epochs. Feel free to print out the model's loss every 10 epochs or so.
8. Use the trained model to predict outputs for `X`. Round the predictions from the model to 0 or 1 using `.round()`.
9. Print out the rounded predictions converted back to a NumPy array using `.numpy()` in the same line. Ensure your code prints the rounded predictions in the exactly the same output format as shown below.
9. Save your program and run it to test the output.
10. To get the flag, run `/challenge/verify <your_file>.py` to verify your solution. You can test your solution by running `/usr/bin/python3 <your_file>.py`

Example output:
```commandline
[[0.]
 [0.]
 [0.]
 [1.]]
```

NOTE: Your output results may differ due to the model randomly initializing its weights at the beginning of training. Running the Python script may take some time due to importing PyTorch.
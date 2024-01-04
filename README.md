# Approximate Nearest Neighbors Algorithm With Triangle Inequality

## Introduction
This project presents an implementation of an Approximate Nearest Neighbors (ANN) algorithm with the incorporation of the triangle inequality. The algorithm is designed for both classification and regression tasks. The key idea behind this implementation is to speed up the nearest neighbors search using the triangle inequality property, thereby reducing computational costs while maintaining reasonable accuracy.

## Mathematics Behind The Algorithm
The algorithm relies on the triangle inequality property, a fundamental concept in metric spaces. In the context of nearest neighbors, the triangle inequality suggests that the direct path between two points is always shorter or equal to the sum of the lengths of two other paths. This property is leveraged to optimize the search for nearest neighbors and enhance the algorithm's efficiency.

## Pseudocode of The Algorithm

### Pseudocode for ANNeighborClassifier

```
1. Initialization: Accept a distance metric during initialization.
2. Training: Store training data and corresponding labels.
3. Prediction: For each new data point:
    - Initialize distances array.
    - Update distances using triangle inequality.
    - Find the index of the minimum distance.
    - Assign the label of the nearest neighbor as the prediction.
4. Scoring: Accept test data, true labels, and a metric.
    - Obtain predicted labels.
    - Evaluate performance using the specified metric.
```

### Pseudocode for ANNeighborRegressor

```
1. Initialization: Accept a distance metric during initialization.
2. Training: Store training data and corresponding target values.
3. Prediction: For each new data point:
    - Initialize distances array.
    - Update distances using triangle inequality.
    - Find the index of the minimum distance.
    - Assign the target value of the nearest neighbor as the prediction.
4. Scoring: Accept test data, true values, and a metric.
    - Obtain predicted values.
    - Evaluate performance using the specified metric.
```


## Modules
The code is organized into several modules for clarity and maintainability:

### Module 1: datasets.py
Handles dataset generation functions, such as make_classification, make_blobs, and make_regression.

### Module 2: plots.py
Contains functions for visualizing data, including 2D scatter plots and line graphs.

### Module 3: metrics.py
Defines evaluation metrics such as accuracy and R-squared.

### Module 4: splitters.py
Implements dataset splitting functions, like splitXy.

### Module 5: neighbors.py
Implements the core functionality of the Approximate Nearest Neighbors Algorithm for both classification and regression tasks.

## Examples
The code includes examples demonstrating how to use the Approximate Nearest Neighbors Algorithm for classification and regression problems.

### Example 1: Nearest Neighbors For Classification Problem
Illustrates the application of the algorithm for a classification task.

### Example 2: Nearest Neighbors For Regression Problem
Demonstrates the use of the algorithm for a regression task.

## Usage
To use the project, make sure to install necessary dependencies by running pip install numpy matplotlib before executing the code in the notebook.

## Contributing
The project welcomes contributions from other users. They can open an issue or submit a pull request with their ideas or changes.

## License
The project is licensed under the terms of the MIT license.






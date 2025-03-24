# Linear Regression from Scratch

## Overview
This section of the repository contains an implementation of **Linear Regression** from scratch using Python and NumPy. The goal is to understand the underlying mathematical concepts and implement the algorithm without relying on high-level machine learning libraries.

## Mathematical Foundations
Linear Regression aims to model the relationship between an independent variable **X** and a dependent variable **y** using a linear equation:

\[ y = wX + b \]

Where:
- **w** (weights) represents the slope of the regression line.
- **b** (bias) represents the intercept.

To optimize **w** and **b**, we minimize the **Mean Squared Error (MSE)** cost function:

\[ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2 \]

where \( \hat{y} \) is the predicted value. The optimization is performed using **Gradient Descent**, which updates weights and bias iteratively:

\[ w := w - \alpha \cdot \frac{1}{n} \sum_{i=1}^{n} X_i (\hat{y_i} - y_i) \]
\[ b := b - \alpha \cdot \frac{1}{n} \sum_{i=1}^{n} (\hat{y_i} - y_i) \]

where \( \alpha \) is the learning rate.

## Implementation Process
1. **Initialize Parameters**: Weights and bias are initialized to zero.
2. **Compute Predictions**: Using the linear equation \( y = wX + b \).
3. **Calculate Gradients**: Compute the derivative of the loss function with respect to \( w \) and \( b \).
4. **Update Parameters**: Adjust \( w \) and \( b \) using gradient descent.
5. **Repeat**: Iterate this process for a predefined number of epochs.

## Thought Process Behind Scratch Implementation
Instead of using Scikit-Learn, implementing Linear Regression from scratch provides:
- **A deeper understanding** of how the model learns from data.
- **Control over hyperparameters**, such as learning rate and number of iterations.
- **Insights into optimization** techniques like Gradient Descent.
- **Debugging skills**, as it helps in understanding issues like slow convergence or improper scaling.

## Results & Evaluation
The implementation uses a synthetic dataset generated via `sklearn.datasets.make_regression()`. After training:
- **Mean Squared Error (MSE)** is computed to assess the model’s performance.
- **R² Score (Coefficient of Determination)** is calculated to measure how well the model fits the data.
- **Visualization**: A scatter plot of training and test data is plotted, along with the regression line.

---

This implementation showcases how machine learning models can be built from scratch, reinforcing core concepts beyond simply calling `sklearn.LinearRegression()`.


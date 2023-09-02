# 1. What is the difference between Parametric and Non Parametric Algorithms?


| Aspect                                | Parametric Algorithms               | Non-Parametric Algorithms            |
|---------------------------------------|------------------------------------|-------------------------------------|
| **Description**                       | Make strong assumptions about data  | Make minimal assumptions about data |
|                                       | distribution and have a fixed      | and do not have a fixed number of   |
|                                       | number of parameters.               | parameters.                         |
| **Examples**                           | Linear Regression,                 | k-Nearest Neighbors (KNN),          |
|                                       | Logistic Regression                | Decision Trees                      |
| **Advantages**                         | Computational efficiency when      | Flexibility to capture complex      |
|                                       | assumptions are met.                | relationships; no strong assumptions|
|                                       |                                    | about data distribution.            |
| **Disadvantages**                     | May yield biased results when      | Prone to overfitting, especially    |
|                                       | assumptions are not met; may not   | with small datasets; potentially   |
|                                       | capture complex, non-linear        | fitting noise in data.              |
| **Example Use Case**                  | Predicting income based on age     | Predicting income based on age     |

**Example**:

*Parametric Approach: Linear Regression*

- Assumption: Income and age have a linear relationship.
- Model: Income = β₀ + β₁ * Age
- The model assumes a straight-line relationship between age and income.

*Non-Parametric Approach: k-Nearest Neighbors (KNN)*

- No assumption about the specific form of the relationship.
- For a new data point with age 'A,' KNN finds the 'k' nearest neighbors in the training data and averages their incomes.

This table provides a concise overview of the differences between parametric and non-parametric algorithms, including descriptions, examples, advantages, disadvantages, and a practical use case. You can easily copy and paste this table into a document editor and further format it as needed.
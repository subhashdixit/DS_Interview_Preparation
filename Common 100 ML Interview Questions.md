# 1. What is the difference between Parametric and Non Parametric Algorithms?
Ans:
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

# 2. Difference between convex and non-convex cost function; what does it mean when a cost function is non-convex?
Ans:
| Aspect                                        | Convex Cost Function                 | Non-Convex Cost Function                |
|-----------------------------------------------|------------------------------------|----------------------------------------|
| **Shape of Cost Function**                    | Forms a convex curve with a single  | Forms a non-convex curve with one      |
|                                               | global minimum.                     | or more local minima.                  |
| **Optimization Property**                     | Easier to optimize as gradient     | More challenging to optimize due to   |
|                                               | descent methods converge to the    | presence of multiple local minima.    |
|                                               | global minimum.                     |                                        |
| **Global vs. Local Minima**                   | Has only one global minimum.       | May have multiple local minima.       |
| **Optimal Solution**                         | The global minimum is the          | The global minimum may not be        |
|                                               | optimal solution.                   | reachable using gradient descent.     |
| **Examples**                                  | Linear Regression with Mean        | Neural networks with non-linear       |
|                                               | Squared Error (MSE) cost function  | activation functions and complex      |
|                                               |                                    | architectures.                        |
| **Impact on Optimization**                   | Efficient optimization;            | Slower convergence, and risk of      |
|                                               | convergence to global minimum.     | getting stuck in local minima.        |
| **What It Means When Non-Convex**            | Cost function is convex when the   | A non-convex cost function implies    |
| Cost Function Is Non-Convex                  | line connecting any two points in  | the presence of multiple potential    |
|                                               | the curve lies above the curve.    | solutions, making optimization more   |
|                                               |                                    | complex and sensitive to initialization.|

**Example**: 

*Convex Cost Function (MSE in Linear Regression)*

- Shape: Forms a convex curve with a single global minimum.
- Optimization: Easier to optimize as gradient descent methods converge to the global minimum.
- Examples: Linear Regression with Mean Squared Error (MSE) cost function.
- Impact: Efficient optimization, with convergence to the global minimum.

*Non-Convex Cost Function (Neural Networks)*

- Shape: Forms a non-convex curve with multiple local minima.
- Optimization: More challenging to optimize due to the presence of multiple local minima.
- Examples: Neural networks with non-linear activation functions and complex architectures.
- Impact: Slower convergence, risk of getting stuck in local minima, and sensitivity to initialization.

This table provides a clear comparison between convex and non-convex cost functions, covering aspects such as shape, optimization properties, global vs. local minima, examples, and their impact on optimization.

# 3. How do you decide when to go for deep learning for a project?
Ans:
| Aspect                                     | Decision Criteria for Deep Learning                           |
|--------------------------------------------|--------------------------------------------------------------|
| **Data Characteristics**                   | - Large and complex datasets.                                |
|                                            | - High-dimensional data, such as images, audio, or text.     |
| **Complex Patterns**                       | - When the problem involves capturing intricate patterns,    |
|                                            |   such as image recognition, natural language processing.    |
|                                            | - Hierarchical feature representations are needed.          |
| **Computational Resources**                | - Access to powerful GPUs or TPUs for training deep models.  |
|                                            | - Sufficient computational resources and memory capacity.    |
| **Task Complexity**                        | - Complex tasks like speech recognition, object detection,   |
|                                            |   or machine translation where deep models excel.           |
| **Data Size vs. Model Complexity Tradeoff** | - When increasing model complexity leads to performance     |
|                                            |   improvements and is justified by the data size.          |
| **Transfer Learning Opportunities**         | - When pre-trained deep learning models (e.g., CNNs, BERT)  |
|                                            |   can be leveraged to bootstrap your project.              |

**Example**: 

Suppose you are working on an image recognition project where you need to identify various objects in images. Deep learning, particularly convolutional neural networks (CNNs), may be a suitable choice due to the following factors:

- You have a large dataset of labeled images.
- The task involves capturing intricate patterns and features within images.
- You have access to GPUs or TPUs for model training.
- The problem complexity justifies the use of deep models.
- Transfer learning with pre-trained CNNs can expedite the development.

This table provides a structured approach to deciding when to opt for deep learning in a project, considering data characteristics, complexity, computational resources, task requirements, data size, and transfer learning opportunities. You can use this as a reference guide to make informed decisions about deep learning adoption in your projects.

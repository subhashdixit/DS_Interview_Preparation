# 1. What is the difference between Parametric and Non Parametric Algorithms?
Ans:

| Aspect                                | Parametric Algorithms               | Non-Parametric Algorithms            |
|---------------------------------------|------------------------------------|-------------------------------------|
| **Description**                       | Make strong assumptions about data distribution and have a fixed number of parameters. | Make minimal assumptions about data and do not have a fixed number of parameters .|
| **Examples**                           | Linear Regression, Logistic Regression | k-Nearest Neighbors (KNN), Decision Trees |
| **Advantages**                         | Computational efficiency when assumptions are met. | Flexibility to capture complex relationships; no strong assumptions about data distribution. |
| **Disadvantages**                     | May yield biased results when assumptions are not met; may not capture complex, non-linear | Prone to overfitting, especially with small datasets; potentially fitting noise in data. |
| **Example Use Case**                  | Predicting income based on age | Predicting income based on age |

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

| Aspect                                | Convex Cost Function             | Non-Convex Cost Function            |
|---------------------------------------|---------------------------------|------------------------------------|
| **Description**                       | Forms a convex shape.            | Does not form a convex shape.      |
| **Shape Example**                     | ![Convex Shape Example](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Convex_polygon_illustration.PNG/220px-Convex_polygon_illustration.PNG) | ![Non-Convex Shape Example](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Concave_polygon_illustration.PNG/220px-Concave_polygon_illustration.PNG) |
| **Convex Cost Function Example**      | Mean Squared Error (MSE) in Linear Regression: J(θ) = (1/2m) Σ(yᵢ - hθ(xᵢ))², where θ represents model parameters. | None provided in the table due to complexity; typically, real-world cost functions exhibit non-convexity. |
| **Non-Convex Cost Function Example**  | Neural Network Loss Function (e.g., Cross-Entropy Loss): J(θ) = -Σ(yᵢ * log(hθ(xᵢ)) + (1 - yᵢ) * log(1 - hθ(xᵢ))), where θ represents neural network weights. |                                    |
| **Meaning of Non-Convexity**          | Multiple local minima; gradient-based optimization may converge to suboptimal solutions. | Multiple local minima and possibly saddle points; optimization can get stuck at suboptimal points. |
| **Practical Implications**            | Optimization is relatively straightforward; global minimum is also the local minimum. | Optimization is challenging; finding the global minimum is not guaranteed. |
| **Use in Machine Learning**            | Often used in linear regression. | Commonly found in neural networks, deep learning, and complex models. |

**Example Use Case**:

- **Convex Cost Function Example (Linear Regression):**
  - Cost Function: Mean Squared Error (MSE)
  - Formula: J(θ) = (1/2m) Σ(yᵢ - hθ(xᵢ))²
  - Convex Shape: ![Convex Shape Example](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Convex_polygon_illustration.PNG/220px-Convex_polygon_illustration.PNG)

- **Non-Convex Cost Function Example (Neural Network):**
  - Cost Function: Cross-Entropy Loss
  - Formula: J(θ) = -Σ(yᵢ * log(hθ(xᵢ)) + (1 - yᵢ) * log(1 - hθ(xᵢ)))
  - Non-Convex Shape: ![Non-Convex Shape Example](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Concave_polygon_illustration.PNG/220px-Concave_polygon_illustration.PNG)

In a convex cost function, the shape is convex, and optimization is relatively straightforward, whereas in a non-convex cost function, the shape is non-convex, leading to challenges in optimization due to multiple local minima and the possibility of getting stuck at suboptimal points.

# 3. How do you decide when to go for deep learning for a project?
Ans:

| Aspect                                | Decision Criteria                  | Numerical Example                   |
|---------------------------------------|------------------------------------|-------------------------------------|
| **Data Size & Complex Patterns**      | Deep learning is beneficial when dealing with large datasets (e.g., millions of data points) and complex data patterns (e.g., intricate features in images or text). | For instance, a project involving 1 million high-resolution images for image classification justifies deep learning due to data size and complexity. |
| **Computational Resources**            | Availability of high-performance hardware (e.g., GPUs) and sufficient computing resources is essential for deep learning projects due to computational intensity. | If you have access to a powerful GPU cluster or cloud resources capable of handling the computational load, deep learning is feasible. |
| **Interpretability & Existing Knowledge**| Deep learning models are often considered black boxes, making interpretation challenging. If interpretability is crucial, consider other models. Familiarity with deep learning frameworks and expertise in training complex neural networks is required for success. | If interpretability is a critical requirement, and you need to explain model decisions, simpler models like decision trees may be preferred over deep learning. However, if your team has prior experience with deep learning and can readily implement models, it can be a suitable choice. |

This consolidated table provides a more concise overview of the decision criteria for choosing deep learning for a project, including data size and complexity, computational resources, interpretability, and existing knowledge. It also includes a numerical example to illustrate the decision-making process.

# 4. Give an example of when False positive is more crucial than false negative and vice versa?
Ans:

| Aspect            | False Positive More Crucial | False Negative More Crucial |
|-------------------|-----------------------------|-----------------------------|
| **Description**   | Occurs when a positive event is incorrectly identified as true, leading to unnecessary actions or consequences. | Occurs when a negative event is incorrectly identified as false, potentially missing a critical event. |
| **Meaning**       | False positives are situations where the system or test wrongly indicates the presence of something that isn't there. | False negatives occur when the system or test fails to identify something that is present. |
| **Numerical Example** | Medical Testing: In disease screening, a false positive result can cause unnecessary stress and treatments. | Security Screening: In airport security, a false negative for a dangerous item poses a significant risk. |
| **Use Case Importance** | Medical Diagnosis, Fraud Detection | Security Screening, Rare Disease Detection |

In scenarios where false positives are more crucial, the focus is on minimizing incorrect positive identifications to avoid unnecessary consequences (e.g., in medical testing). Conversely, when false negatives are more crucial, the priority is on reducing instances where important events are missed (e.g., in security screening).

# 5. Why is “Naive” Bayes naive?
Ans:

| Aspect                                    | Explanation                                                                   |
|-------------------------------------------|-------------------------------------------------------------------------------|
| **Why "Naive" Bayes Is Naive**             | The term "Naive" in Naive Bayes refers to the simplifying assumption that features are conditionally independent given the class. In other words, it assumes that the presence or absence of one feature doesn't affect the presence or absence of another feature, which is often overly simplistic and rarely holds true in real-world data. This simplification is made for computational efficiency and ease of calculation but may not reflect the actual dependencies between features in a dataset.  |

**Example**:

Suppose we want to classify emails as spam or not based on two features: the presence of the word "free" (F) and the presence of the word "money" (M). The "naive" assumption is that the occurrence of "free" and "money" in an email is independent, given whether it's spam or not.

Using Bayes' theorem:
\[P(Spam | F, M) \propto P(F | Spam) \cdot P(M | Spam) \cdot P(Spam)\]
\[P(Not Spam | F, M) \propto P(F | Not Spam) \cdot P(M | Not Spam) \cdot P(Not Spam)\]

The assumption that \(P(F, M | Spam) = P(F | Spam) \cdot P(M | Spam)\) and \(P(F, M | Not Spam) = P(F | Not Spam) \cdot P(M | Not Spam)\) simplifies the calculation. However, in practice, it's unlikely that the presence of "free" and "money" is entirely independent in spam emails, making the "naive" assumption a simplification.

# 6. Give an example where the median is a better measure than the mean?
Ans:
Certainly! Here's the answer in the previous table format, including numerical examples:

| Aspect                                | Median                                        | Mean                                         |
|---------------------------------------|-----------------------------------------------|----------------------------------------------|
| **Definition**                         | The median is the middle value in a dataset when it's sorted, separating the higher half from the lower half.  | The mean (average) is the sum of all values divided by the total number of values.        |
| **Use Case Example**                  | **Example 1: Household Incomes**           | **Example 2: Exam Scores**                   |
|                                       | Consider a dataset of household incomes where there are a few extremely high-income earners (outliers). | In a class of students, you want to understand the average exam score.                  |
|                                       | Household Incomes: $30,000, $35,000, $40,000, $42,000, $50,000, $250,000               | Exam Scores: 85, 88, 90, 92, 94, 56, 58, 59, 60, 100                           |
| **Advantages**                         | Robust to outliers; not heavily influenced by extreme values.                             | Sensitive to extreme values; reflects the overall distribution.                         |
| **Disadvantages**                     | May not represent the central tendency if the data is skewed or has outliers.             | Can be affected by outliers, making it less robust.                                      |
| **When Median is Preferred**           | **Example 1:** When assessing the typical income of households, especially with significant income disparities, the median is preferred to avoid being skewed by a few exceptionally high earners. | **Example 2:** When analyzing exam scores in a class, particularly if a few students scored exceptionally high or low, the median provides a more representative measure of the typical student's performance. |
| **Calculation**                        | Median Calculation: Arrange the incomes in ascending order and select the middle value (or the average of the two middle values in case of an even number of data points). | Mean Calculation: Sum of all exam scores divided by the total number of students (Sum / Number of Students). |

**Example 1 (Median):**

For the household incomes example:

1. Sort the incomes in ascending order: $30,000, $35,000, $40,000, $42,000, $50,000, $250,000.
2. The median is the middle value, which is $42,000.
3. The median represents the typical income better than the mean, which would be significantly affected by the high-income outlier of $250,000.

**Example 2 (Median):**

For the exam scores example:

1. Sort the exam scores in ascending order: 56, 58, 59, 60, 85, 88, 90, 92, 94, 100.
2. The median is the middle value, which is 88.
3. The median is a more robust measure of typical performance, especially when there are outliers like the score of 100.

In both examples, the median provides a better measure of central tendency in the presence of outliers or skewed data compared to the mean.

# 7. What do you mean by the unreasonable effectiveness of data?
Ans:


| Aspect                                | Unreasonable Effectiveness of Data and          | Comparison of DL and ML Performance  |
|---------------------------------------|--------------------------------------------------|-------------------------------------|
| **Definition**                        | Refers to the phenomenon where having more data | Deep Learning (DL) typically requires large amounts of data for its complex models, while Machine Learning (ML) can work effectively with smaller datasets. |
| **Explanation**                       | With abundant data, models can learn diverse and intricate patterns, reducing overfitting. ML models may plateau in performance due to limited data, while DL models can continue improving with more data. | More data often results in better model performance, especially in complex DL models. ML models may plateau with limited data, and DL models can continue to benefit from more data. |
| **Numerical Example** (Hypothetical)  | Suppose you're building a spam email classifier. With a small dataset of 1,000 emails, your ML classifier achieves 85% accuracy. When you acquire a larger labeled dataset of 100,000 emails, your DL model achieves 95% accuracy. | In a hypothetical example, a spam email classifier achieves 85% accuracy with a small dataset of 1,000 emails, but the accuracy improves to 95% when using a larger dataset of 100,000 emails. |
| **Comparison Conclusion**              | More data often results in better model performance, especially in complex DL models. ML models may plateau with limited data, and DL models can continue to benefit from more data. | DL outperforms ML when ample data is available, but ML can be more resource-efficient with smaller datasets and simpler algorithms. |

**Explanation**:

The "Unreasonable Effectiveness of Data" refers to the concept that having more data can significantly improve model performance, reducing overfitting and allowing models to learn intricate patterns. In a hypothetical example, a spam email classifier achieves 85% accuracy with a small dataset of 1,000 emails, but the accuracy improves to 95% when using a larger dataset of 100,000 emails. This highlights that Deep Learning (DL) models with millions of parameters can excel with extensive data, achieving state-of-the-art results. In comparison, Machine Learning (ML) models may plateau in performance with limited data and can be resource-efficient with smaller datasets and simpler algorithms.

# 8. Why KNN is known as a lazy learning technique?
Ans:

| Aspect                        | K-Nearest Neighbors (KNN)                            |
|-------------------------------|------------------------------------------------------|
| **Lazy Learning Technique**   | KNN is known as a lazy learning technique because it defers the model's learning until prediction time, making minimal assumptions during training. |
| **Description**               | It classifies or predicts based on the majority class or average of the 'k' nearest neighbors in the training data. |
| **Example**                   | Let's say we have a dataset of flowers with features like petal length and width. When we want to classify a new flower, KNN finds the 'k' training examples with the most similar feature values (nearest neighbors) and assigns the majority class among them to the new flower. |
| **Advantages**                 | - Simplicity in implementation. - Ability to capture complex decision boundaries. - No need to retrain the model when new data arrives. |
| **Disadvantages**             | - Computationally expensive for large datasets. - Sensitive to the choice of 'k.' - Prone to noise and outliers. |
| **Use Cases**                 | - Image recognition. - Recommender systems. - Anomaly detection. - Handwriting recognition. - Medical diagnosis. |

KNN is referred to as a lazy learning technique because it doesn't generalize during training; it stores the entire training dataset and only performs computations when making predictions, considering the nearest neighbors.

# 9. What do you mean by semi supervised learning?
Ans: 

| Aspect                    | Semi-Supervised Learning                                                  |
|---------------------------|-----------------------------------------------------------------------------|
| **Definition**            | Semi-supervised learning is a machine learning paradigm that combines both labeled and unlabeled data in the training process. |
| **Key Idea**               | Utilizes a combination of limited labeled data and a larger amount of unlabeled data to improve model performance. |
| **Example Scenario**       | Suppose you have a dataset of images with some images labeled as "cats" and "dogs" (labeled data) and a larger set of unlabeled images. |
| **Benefits**               | - Cost-effective as labeling data is often expensive and time-consuming. <br> - Can boost model performance when labeled data is scarce. |
| **Challenges**             | - Requires a reliable method for incorporating unlabeled data effectively. <br> - Performance heavily depends on the quality of the unlabeled data. |
| **Use Case Example**       | In image classification, with limited labeled examples of cat and dog images, semi-supervised learning can leverage a large pool of unlabeled images to improve classification accuracy. |

**Example Numerical Scenario:**

Suppose you have 100 labeled images where 50 are labeled as "cat" and 50 as "dog." You also have an additional 9000 unlabeled images. In semi-supervised learning, you can use this combination of 100 labeled and 9000 unlabeled images to train a more accurate image classification model compared to using only the 100 labeled images.

This table provides a concise overview of semi-supervised learning, including its definition, key idea, benefits, challenges, use case example, and a numerical scenario to illustrate the concept.

# 10. What is an OOB error and how is it useful?
Ans:

| Aspect                  | Out-of-Bag (OOB) Error                                           |
|-------------------------|-----------------------------------------------------------------|
| **Description**         | OOB error is a metric used in the context of bagging algorithms like Random Forest. It quantifies the model's prediction error on the data points that were not used in a particular bootstrap sample. |
| **Calculation**         | Calculate the prediction error for each data point using only the trees in the Random Forest ensemble that didn't include that data point in their bootstrap sample. |
| **Usefulness**          | OOB error serves as a reliable estimate of a model's performance without the need for a separate validation set, making it useful for assessing model accuracy and preventing overfitting. |
| **Example**             | Suppose we have a Random Forest with 100 decision trees. For each data point, the model calculates predictions based on the votes of the trees that didn't use that data point during training. The OOB error is then the average prediction error across all data points. |
| **Advantages**          | - Provides a robust estimate of model performance.
                         | - Eliminates the need for a separate validation set, saving data and simplifying the modeling process. |
| **Disadvantages**       | - May be computationally intensive with a large number of trees.
                         | - OOB error is an estimate and may have some variability. |

**Explanation**:

Out-of-Bag (OOB) error is a metric used in bagging algorithms like Random Forest. It calculates the prediction error for each data point based on the votes of the decision trees in the ensemble that did not include that data point in their bootstrap sample during training. The OOB error serves as a reliable estimate of the model's performance without the need for a separate validation set, making it useful for assessing model accuracy and preventing overfitting. For example, in a Random Forest with 100 decision trees, the OOB error is calculated as the average prediction error across all data points. While OOB error simplifies the modeling process and provides a robust estimate, it can be computationally intensive with a large number of trees and may have some variability due to its estimation nature.

# 11. In what scenario decision tree should be preferred over random forest?
Ans:

| Scenario                                     | Decision Tree                                   | Random Forest                                  |
|----------------------------------------------|-------------------------------------------------|------------------------------------------------|
| **When to Prefer**                           | - Decision trees can be preferred when          | - Random forests are preferred when you want   |
|                                              | you prioritize model interpretability.           | improved predictive performance and robustness.  |
|                                              | - Simple and transparent models are needed.     | - Handling complex, non-linear relationships    |
|                                              | - Quick insights into feature importance are     | and capturing interactions between features     |
|                                              |   essential.                                    | is critical.                                    |
| **Example:**                                 | Imagine a loan approval scenario where you     | In a medical diagnosis task, where you have a  |
|                                              | need to explain the decision process to         | large dataset of patient data with many        |
|                                              | customers. A decision tree can provide clear   | features. Random forests can handle the         |
|                                              | criteria for loan approval, which is easy to   | complexity and provide robust predictions.     |

**Numerical Example:**

Let's consider a simplified binary classification problem. We have a dataset of customer information, and we want to predict whether a customer will purchase a product (1) or not (0) based on two features: age and income.

- **Decision Tree:** A decision tree might split the data based on age and income, creating a simple tree structure. For example:
  ```
  If Age <= 30 and Income <= $50,000, Predict: 0
  Else, Predict: 1
  ```

- **Random Forest:** Random forests consist of multiple decision trees. Each tree in the forest might make different splits, and the final prediction is based on a majority vote or averaging of individual tree predictions. For example, if we have three decision trees, and they make predictions as follows:
  ```
  Tree 1: Predict: 0
  Tree 2: Predict: 1
  Tree 3: Predict: 1
  ```

  The random forest may predict the majority class, which is 1, as the final prediction.

In this scenario, if you prioritize simplicity and interpretability, you may prefer the decision tree. However, if you aim for improved predictive accuracy and handling complex relationships, a random forest might be preferred.

This table format provides a clear comparison between decision trees and random forests in different scenarios, along with a numerical example to illustrate the concept.

# 12. Why Logistic Regression is called regression?
Ans:
**Description/Explanation:**

- **Logistic Regression** is a classification algorithm despite its name because it predicts a binary outcome (0 or 1).
- The term "regression" in its name is a historical artifact, referring to the logistic function used in the algorithm.

**Numerical Examples:**

- In a binary classification problem where we predict whether an email is spam (1) or not spam (0), logistic regression might output a probability like 0.75, indicating a 75% chance that the email is spam. This is not a continuous numeric value but a probability used for classification.
- Logistic regression uses the logistic (sigmoid) function, which maps any real-valued number to a value between 0 and 1, making it suitable for classification tasks despite the term "regression" in its name.
- 
# 13. What is Online Machine Learning? How is it different from Offline machine learning? List some of it’s applications?
Ans:
**Description/Explanation:**

**Online Machine Learning:**
Online machine learning, also known as incremental or streaming machine learning, is a machine learning paradigm that involves training models on continuously arriving data. Unlike traditional offline machine learning, where models are trained on fixed datasets, online learning adapts to new data as it becomes available. 

**Offline Machine Learning:**
Offline machine learning, or batch learning, involves training models on a fixed dataset and updating them periodically when new data is collected. Models are trained from scratch each time with the entire dataset.

**Differences:**

- **Data Handling:**
  - *Online Machine Learning:* Handles data in a continuous stream, updating models on the fly.
  - *Offline Machine Learning:* Trains models on a static dataset.

- **Training Frequency:**
  - *Online Machine Learning:* Continuous and incremental model updates.
  - *Offline Machine Learning:* Periodic model retraining.

- **Resource Usage:**
  - *Online Machine Learning:* Requires fewer computational resources per update.
  - *Offline Machine Learning:* Typically requires more computational resources during batch training.

- **Applications:**
  - *Online Machine Learning:* Suited for applications with dynamic data and real-time decision-making.
  - *Offline Machine Learning:* Typically used for batch data analysis and modeling.

**Numerical Examples:**

**Online Machine Learning:**
Imagine a recommendation system for an e-commerce website. It continuously collects user behavior data (clicks, purchases) and updates the recommendation model in real-time as users interact with the platform. This allows the system to adapt to changing user preferences immediately.

**Offline Machine Learning:**
Consider a healthcare system that periodically analyzes patient data to predict disease outcomes. The system collects data for a fixed period, such as a month, and then retrains predictive models using this static dataset. The models are not updated until the next batch of data is available.

**Applications:**

- **Online Machine Learning Applications:**
  - Real-time recommendation systems (e.g., e-commerce).
  - Fraud detection in financial transactions.
  - Sentiment analysis of live social media data.
  - Predictive maintenance in manufacturing.
  - Adaptive game AI in gaming.

- **Offline Machine Learning Applications:**
  - Batch analysis of historical sales data for demand forecasting.
  - Training deep learning models on large image datasets.
  - Analyzing customer churn based on quarterly data.
  - Annual financial reporting and forecasting.
  - Conducting research studies on fixed datasets.

Online machine learning is valuable in scenarios where data arrives continuously and immediate decision-making or adaptation is required. In contrast, offline machine learning is suitable for scenarios where data is collected in batches and periodic model updates are acceptable.

# 14. What is No Free Lunch Theorem?
Ans:
**Description/Explanation:**

- The No Free Lunch Theorem (NFLT) is a fundamental concept in machine learning.
- It suggests that there is no one-size-fits-all algorithm or model that performs best for all types of problems.
- NFLT implies that the performance of any machine learning algorithm is highly dependent on the specific characteristics and assumptions of the problem it's applied to.

**Numerical Examples:**

1. Suppose you have a classification problem where the data is linearly separable. In this case, a linear classifier like Logistic Regression may perform very well. However, if you apply a highly non-linear model like a deep neural network without proper data preprocessing, its performance may be inferior.

2. Conversely, consider a problem where the data exhibits complex, non-linear relationships. Here, a decision tree or a random forest might outperform a simple linear model because they can capture intricate patterns in the data.

3. NFLT also applies to optimization algorithms. For example, gradient descent may work well for convex cost functions but struggle to find the global minimum in non-convex functions, where other optimization techniques like genetic algorithms or simulated annealing might be more suitable.

In essence, the No Free Lunch Theorem underscores the importance of selecting the right algorithm or model based on the characteristics and requirements of the specific problem you are trying to solve.

# 15. Imagine you are woking with a laptop of 2GB RAM, how would you process a dataset of 10GB?
# 16.  What are the main differences between Structured and Unstructured Data?
# 17. What are the main points of difference between Bagging and Boosting?
# 18. What are the assumptions of linear regression?
# 19. How do you measure the accuracy of a Clustering Algorithm?
# 20. What is Matrix Factorization and where is it used in Machine Learning?
# 21. What is an Imbalanced Dataset and how can one deal with this problem?
# 22. How do you measure the accuracy of a recommendation engine?
# 23. What are some ways to make your model more robust to outliers?
# 24. How can you measure the performance of a dimensionality reduction algorithm on your dataset?
# 25. What is Data Leakage? List some ways using which you can overcome this problem?
# 26. What is Multicollinearity? How to detect it? List some techniques to overcome Multicollinearity?
# 27. List some ways using which you can reduce overfitting in a model?
# 28. What are the different types of bias in Machine Learning?
# 29. How do you approach a categorical feature with high cardinality?
# 30. Explain Pruning in Decision Trees and how it is done?
# 31.  What is ROC-AUC curve? List some of it’s benefits?
# 32. What are kernels in SVM? Can you list some popular SVM kernels?
# 33. What is the difference between Gini Impurity and Entropy? Which one is better and why?
# 34. Why does L2 regularization give sparse coefficients?
# 35. List some ways using which you can improve a model’s performance.
# 36. Can PCA be used to reduce the dimensionality of a highly nonlinear dataset?
# 37. What’s the difference between probability and likelihood?
# 38. What cross-validation technique would you use on a time series data set?
# 39. Once a dataset’s dimensionality has been reduced, is it possible to reverse the operation? If so, how? If not, why?
# 40. Why do we always need the intercept term in a regression model??
# 41. When Your Dataset Is Suffering From High Variance, How Would You Handle It?
# 42. Which Among These Is More Important Model Accuracy Or Model Performance?
# 43. What is active learning and where is it useful?
# 44. Why is Ridge Regression called Ridge?
# 45. State the differences between causality and correlation?
# 46. Does it make any sense to chain two different dimensionality reduction algorithms?
# 47. Is it possible to speed up training of a bagging ensemble by distributing it across multiple servers?
# 48. If a Decision Tree is underfitting the training set, is it a good idea to try scaling the input features?
# 49. Say you trained an SVM classifier with an RBF kernel. It seems to underfit the training set: should you increase or decrease γ (gamma)? What about C?
# 50. What is cross validation and it's types?
# 51. How do we interpret weights in linear models?
# 52. Which Gradient Descent algorithm (among those we discussed) will reach the vicinity of the optimal solution the fastest? Which will actually converge?
# 53. Why is it important to scale the inputs when using SVMs?
# 54. What is p value and why is it important?
# 55. What is OvR and OvO for multiclass classification and which machine learning algorithm supports this?
# 56. How will you do feature selection using Lasso Regression?
# 57. What is the difference between loss function and cost function?
# 58. What are the common ways to handle missing data in a dataset?
# 59. What is the difference between standard scaler and minmax scaler? What you will do if there is a categorical variable?
# 60. What types of model tend to overfit?
# 61. What are some advantages and Disadvantages of regression models and tree based models?
# 62. What are some important hyperparameters for XGBOOST?
# 63. Can you tell the complete life cycle of a data science project?
# 64. What are the properties of a good ML model?
# 65. What are the different evaluation metrices for a regression model?
# 66. What are the different evaluation metrices for a classification model?
# 67. Difference between R2 and adjusted R2? Why do you preffer adjusted r2?
# 68. List some of the drawbacks of a Linear model
# 69. What do you mean by Curse of Dimensionality?
# 70. What do you mean by Bias variance tradeoff?
# 71. Explain Kernel trick in SVM?
# 72. What is the main difference between Machine Learning and Data Mining?
# 73. Why sometimes it is needed to scale or normalise features?
# 74. What is the difference between Type 1 and Type 2 error?
# 75. What is the difference between a Generative model vs a Discriminative model?
# 76. Why binary_crossentropy and categorical_crossentropy give different performances for the same problem?
# 77. Why does one hot encoding improve machine learning performance?
# 78. Considering the long list of machine learning algorithm, given a data set, how do you decide which one to use?
# 79. Differentiate between wide and tall data formats?
# 80. What is the difference between inductive machine learning and deductive machine learning?
# 81. How will you know which machine learning algorithm to choose for your classification problem?
# 82. What is the difference between Covariance and Correlation?
# 83. How will you find the correlation between a categorical variable and a continuous variable?
# 84. What are the differences between “Bayesian” and “Frequentist” approach for Machine Learning?
# 85. What is the difference between stochastic gradient descent (SGD) and gradient descent ?
# 86. What is the difference between Gaussian Mixture Model and K-Means Algorithm?
# 87. Is more data always better?
# 88. How can you determine which features are the most im- portant in your model?
# 89. Which hyper-parameter tuning strategies (in general) do you know?
# 90. How to select K for K-means?
# 91. Describe the differences between and use cases for box plots and histograms?
# 92. How would you differentiate between Multilabel and MultiClass classification?
# 93. What is KL divergence, how would you define its usecase in ML?
# 94. Can you define the concept of Undersampling and Oversampling?
# 95. Considering a Long List of Machine Learning Algorithms, given a Data Set, How Do You Decide Which One to Use?
# 96. Explain the difference between Normalization and Standardization?
# 97. List the most popular distribution curves along with scenarios where you will use them in an algorithm?
# 98. List all types of popular recommendation systems?
# 99. Which metrics can be used to measure correlation of categorical data?
# 100. Which type of sampling is better for a classification model and why?

- The problem complexity justifies the use of deep models.
- Transfer learning with pre-trained CNNs can expedite the development.

This table provides a structured approach to deciding when to opt for deep learning in a project, considering data characteristics, complexity, computational resources, task requirements, data size, and transfer learning opportunities. You can use this as a reference guide to make informed decisions about deep learning adoption in your projects.

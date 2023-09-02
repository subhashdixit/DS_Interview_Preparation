# 1. What is the difference between Parametric and Non Parametric Algorithms?
Ans:
| Aspect                                | Parametric Algorithms               | Non-Parametric Algorithms            |
|---------------------------------------|------------------------------------|-------------------------------------|
| **Description**                       | Make strong assumptions about data distribution and have a fixed number of parameters. | Make minimal assumptions about data and do not have a fixed number of parameters.|
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

# 4. Give an example of when False positive is more crucial than false negative and vice versa?
Ans:

| Scenario                                     | When False Positive is More Crucial            | When False Negative is More Crucial          |
|----------------------------------------------|----------------------------------------------|---------------------------------------------|
| **Description**                              | In some situations, the cost or impact of    | In other situations, the cost or impact of  |
|                                              | a false positive (Type I Error) can be      | a false negative (Type II Error) can be    |
|                                              | more severe or undesirable than that of a   | more severe or undesirable than that of a   |
|                                              | false negative (Type II Error).             | false positive (Type I Error).             |
| **Example**                                  | 1. **Medical Testing:**                    | 1. **Security Screening:**                |
|                                              |    - False positive in a medical test      |    - False negative in a security         |
|                                              |      for a severe disease might lead to    |      screening at an airport could miss   |
|                                              |      unnecessary treatments, anxiety, and  |      a dangerous item, posing a security |
|                                              |      medical costs.                       |      risk.                               |
|                                              | 2. **Spam Email Detection:**               | 2. **Criminal Justice:**                 |
|                                              |    - Flagging legitimate emails as spam   |    - Releasing a guilty criminal due to  |
|                                              |      (false positive) can result in      |      insufficient evidence (false        |
|                                              |      important messages being missed.    |      negative) can have serious           |
|                                              |                                           |      consequences for public safety.      |

**False Positive More Crucial Scenario**:
In some contexts, such as medical testing and spam email detection, false positives can have significant consequences. For example, a false positive in a medical test might lead to unnecessary treatments, while marking legitimate emails as spam (false positive) can result in important messages being missed.

**False Negative More Crucial Scenario**:
In other scenarios like security screening at an airport and criminal justice, false negatives can be more critical. Missing a dangerous item in security screening (false negative) poses a security risk, and releasing a guilty criminal due to insufficient evidence (false negative) can have serious consequences for public safety.

These scenarios highlight the importance of considering the specific context and consequences when deciding whether false positives or false negatives are more crucial to minimize.

# 5. Why is “Naive” Bayes naive?
Ans:

| Aspect                       | Naive Bayes                 |
|------------------------------|-----------------------------|
| **Name Explanation**         | "Naive" in Naive Bayes     |
|                              | signifies the simplifying   |
|                              | assumption made by the     |
|                              | algorithm.                  |
| **Explanation**              | Naive Bayes is considered  |
|                              | "naive" because it assumes  |
|                              | that all features are      |
|                              | independent and have no    |
|                              | correlation with each other.|
| **Assumption**               | Assumes strong feature     |
|                              | independence, which is     |
|                              | often not true in real-world|
|                              | data.                       |
| **Impact**                   | Despite this simplification,|
|                              | Naive Bayes can perform    |
|                              | surprisingly well in many  |
|                              | classification tasks.       |

**Description**:

- The term "Naive" in "Naive Bayes" refers to a simplifying assumption that the algorithm makes about the independence of features. It assumes that all features used to describe an instance are mutually independent and have no correlation with each other.

**Explanation**:

- Naive Bayes is considered "naive" because it simplifies the modeling process by assuming strong feature independence. In practice, many real-world datasets do not meet this assumption, as features often have complex dependencies and correlations. Despite this simplification, Naive Bayes can perform surprisingly well in many classification tasks, especially when the independence assumption is approximately met.

The "naive" aspect of Naive Bayes highlights its simplicity and the simplifying assumption it relies on, which makes it computationally efficient and easy to implement.

# 6. Give an example where the median is a better measure than the mean
Ans:

| Aspect                    | Mean                                         | Median                                      |
|---------------------------|----------------------------------------------|---------------------------------------------|
| **Definition**            | The mean, or average, is the sum of all values divided by the number of values.   | The median is the middle value in a sorted list of values, where half the values are above and half are below it. |
| **Use Case**              | When dealing with a symmetric or normally distributed dataset.                  | When dealing with skewed or non-normally distributed datasets with outliers. |
| **Example Scenario**      | Calculating the average income in a group of individuals.                    | Examining household income in a city, where a few high-income households skew the data. |
| **Robustness to Outliers** | Sensitive to outliers; outliers can significantly impact the mean.             | More robust to outliers; outliers have less impact on the median. |
| **Summary**               | Appropriate for balanced data distributions.                                   | Preferred for data distributions with skewness or outliers. |

**Example Use Case**:

Suppose you are analyzing the incomes of individuals in two neighborhoods, Neighborhood A and Neighborhood B:

- **Neighborhood A:** Most people have similar incomes, with a few earning exceptionally high salaries.
- **Neighborhood B:** Incomes vary widely, but there are no extremely high earners.

In this scenario, the mean income would likely be higher in Neighborhood A due to the influence of the high earners, whereas the median would better represent the typical income for both neighborhoods. Therefore, the median is a better measure than the mean when dealing with income data that has outliers or a skewed distribution.

# 7. What do you mean by the unreasonable effectiveness of data?
Ans:

| Aspect                                | Explanation                           |
|---------------------------------------|---------------------------------------|
| **Explanation**                        | "The Unreasonable Effectiveness of Data" refers to the phenomenon where having a large and diverse dataset can significantly improve the performance of machine learning models, often surpassing the expectations of model complexity and algorithm sophistication. In essence, it highlights the power of data over the intricacies of algorithms.                                 |
| **Numerical Example**                  | Consider a sentiment analysis task where you're building a machine learning model to classify movie reviews as positive or negative based on their text. If you have a small dataset of 100 reviews, a highly complex model may not perform well due to limited data. However, if you collect a massive dataset of 100,000 reviews, even a simple model like logistic regression can achieve impressive accuracy because it has a wealth of data to learn from. This showcases the "unreasonable effectiveness" of having more data.                                  |

This table provides an explanation of "The Unreasonable Effectiveness of Data" along with a numerical example to illustrate the concept. You can copy and paste this table into a document editor for reference or presentation.

# 8. Why KNN is known as a lazy learning technique?
Ans:

| Aspect                                   | Lazy Learning (KNN)               | Eager Learning (e.g., Decision Trees) |
|------------------------------------------|-----------------------------------|---------------------------------------|
| **Learning Strategy**                    | Learns from the entire training    | Learns a model during training        |
|                                          | dataset during prediction.         |                                       |
| **Computation of Predictions**            | Computes predictions "on the fly"  | Uses a pre-built model to make        |
|                                          | based on nearest neighbors.        | predictions.                          |
| **Storage of Training Data**             | Stores the entire training dataset| Does not require storage of the entire|
|                                          | for prediction.                   | training data after training.         |
| **Example**                               | Suppose we have a dataset of      | In a decision tree, the model is     |
|                                          | images labeled as "cat" or "dog." | constructed based on features like   |
|                                          | When we want to classify a new   | color, size, and shape. For example, |
|                                          | image, KNN finds the 'k' nearest | a decision tree may learn that      |
|                                          | images in the training dataset   | animals with fur and a tail are more |
|                                          | that are similar to the new      | likely to be "cats."                 |
|                                          | image and assigns a class based |                                       |
|                                          | on the majority class among     |                                       |
|                                          | those neighbors.                  |                                       |

**Explanation:**

- **Learning Strategy:** KNN is known as a "lazy learning" or "instance-based learning" technique because it defers learning until the prediction phase. It doesn't build an explicit model during training; instead, it stores the training data and performs computation "on the fly" when making predictions. In contrast, eager learning algorithms, like decision trees, construct a model during training and use it for predictions without retaining the entire training dataset.

**Numerical Example:**

Let's consider an example where we have a dataset of 100 images of animals labeled as "cat" or "dog." We want to classify a new image of an animal.

- **K-Nearest Neighbors (KNN):** 
  - During training, KNN stores all 100 images and their labels.
  - When we want to classify a new image, KNN calculates the similarity between the new image and the 100 stored images by comparing their features (e.g., pixel values).
  - It selects the 'k' (e.g., 5) nearest images in the training dataset based on similarity.
  - KNN counts the majority class among these 'k' neighbors and assigns it as the predicted class for the new image.

- **Eager Learning (Decision Trees):**
  - Decision trees build a model during training based on features like color, size, and shape.
  - The model may learn rules such as "if an animal has fur and a tail, classify it as a 'cat.'"
  - During prediction, the model is used to classify the new image based on these learned rules without storing the entire training dataset.

In summary, KNN is known as a lazy learning technique because it retains and uses the entire training dataset for predictions, performing computations dynamically based on the nearest neighbors, whereas eager learning methods build a fixed model during training and use it for predictions.

# 9. What do you mean by semi supervised learning?
Ans: 

| Aspect                                  | Semi-Supervised Learning                                          |
|-----------------------------------------|-------------------------------------------------------------------|
| **Definition**                           | Semi-supervised learning is a machine learning paradigm that combines both labeled and unlabeled data to train models. It leverages the availability of a small amount of labeled data and a larger pool of unlabeled data.                        |
|                                         |                                                                   |
| **Numerical Example**                   | Imagine you have a dataset of customer reviews for a product. Only a small fraction of the reviews are labeled as positive (+) or negative (-), and the rest are unlabeled.                               |
|                                         |                                                                   |
| **How It Works**                        | - A small labeled dataset is used to train an initial model.     |
|                                         | - The initial model is then used to make predictions on the unlabeled data.                                                                                     |
|                                         | - The predicted labels for the unlabeled data are combined with the small labeled dataset, forming a larger labeled dataset.        |
|                                         | - The model is retrained on this larger labeled dataset to improve its accuracy.                                        |
| **Advantages**                           | - Utilizes unlabeled data, which is often abundant, leading to potentially more accurate models.                       |
|                                         | - Reduces the cost and effort of manual labeling since only a small portion of the data needs to be labeled.                   |
| **Disadvantages**                       | - The quality of predictions on unlabeled data can affect the overall model's performance.                           |
|                                         | - The success of semi-supervised learning depends on the assumption that the unlabeled data distribution is similar to the labeled data.                        |
| **Use Case**                             | Classifying customer reviews as positive or negative using a combination of a few labeled reviews and a large pool of unlabeled reviews.                                              |

**Numerical Example Explanation**:

In this example, semi-supervised learning is applied to a customer review dataset. Only a small fraction of the reviews are labeled as positive (+) or negative (-), while the majority of the reviews remain unlabeled. The process involves:

1. Training an initial model using the small labeled dataset.
2. Using the initial model to predict labels for the unlabeled reviews.
3. Combining the predicted labels with the small labeled dataset to create a larger labeled dataset.
4. Retraining the model on this expanded dataset to improve its accuracy in classifying reviews as positive or negative.

This approach reduces the need for manually labeling a large number of reviews, making it more cost-effective and efficient. However, the success of semi-supervised learning relies on the assumption that the distribution of unlabeled data is similar to the labeled data.

# 10. What is an OOB error and how is it useful?
Ans:

| Aspect                | OOB Error (Out-of-Bag Error)              |
|-----------------------|------------------------------------------|
| **Description**       | OOB error is an estimate of a model's    |
|                       | prediction error using data points that  |
|                       | were not included in the bootstrap      |
|                       | training sample.                         |
| **Calculation**       | 1. For each data point in the training   |
|                       |    dataset, check if it was included in |
|                       |    the bootstrap sample (about 63.2% of |
|                       |    the data on average).                |
|                       | 2. If a data point was not included,    |
|                       |    use it for OOB evaluation.           |
|                       | 3. Calculate the prediction error for   |
|                       |    each OOB data point.                 |
| **Usefulness**        | OOB error provides an unbiased estimate |
|                       | of a model's performance without the   |
|                       | need for a separate validation set.     |
|                       | It's particularly useful for assessing  |
|                       | the accuracy of random forest models.  |
| **Example**           | Suppose you have a dataset of 100      |
|                       | data points. You create a random       |
|                       | forest with 500 trees using bootstrap  |
|                       | sampling. Each tree is trained on a    |
|                       | random sample of the data, and on      |
|                       | average, about 63.2% of the data points|
|                       | are included in each bootstrap sample. |
|                       |                                           |
|                       | To calculate OOB error:                |
|                       | 1. For each data point, check if it was |
|                       |    included in the training sample of  |
|                       |    each tree. If not, use it for OOB   |
|                       |    evaluation.                         |
|                       | 2. Calculate prediction error for each |
|                       |    OOB data point across all 500 trees.|
|                       | 3. OOB error is the average of these   |
|                       |    prediction errors.                  |

**Explanation:**

- OOB error is a valuable metric for random forest models because it allows you to estimate model performance without the need for a separate validation set. It leverages the fact that each tree in the forest is trained on a different subset of data, and the remaining data points (out-of-bag samples) can be used to estimate how well the model generalizes to unseen data.

- By calculating OOB error for each data point across all trees and averaging them, you obtain an unbiased estimate of the model's prediction error. This estimate is useful for model selection, hyperparameter tuning, and assessing the overall quality of the random forest model.

In the example provided, OOB error is calculated for a random forest with 500 trees using a dataset of 100 data points. Each tree is trained on a different bootstrap sample, and the OOB data points are used to estimate the model's prediction error. The OOB error is the average of these errors across all trees.

This metric is especially beneficial when working with ensemble methods like random forests, where aggregating predictions from multiple trees is common.

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

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

| Scenario                                         | Decision Tree                              | Random Forest                             |
|--------------------------------------------------|--------------------------------------------|--------------------------------------------|
| **When to Prefer:**                               | - When interpretability is crucial, and you need a single, understandable tree.                            | - When you seek higher predictive accuracy and robustness to outliers or noisy data.                      |
| **Example Use Case:**                          | Medical diagnosis with simple, explainable rules:      | Predicting customer churn in a telecom company with a large dataset of diverse features:           |
| **Description:**                                     | Decision trees provide a clear, understandable decision path, which can be critical in scenarios where interpretability is more important than marginal gains in accuracy.   | Random Forest combines multiple decision trees, reducing overfitting and improving generalization performance, making it suitable for complex, high-dimensional data.   |
| **Advantages:**                                    | - Easy to visualize and explain.                 | - Reduces overfitting through ensemble learning.                                                     |
|                                                            | - Works well with small to medium-sized datasets. | - Captures complex relationships in data.                                                               |
| **Disadvantages:**                               | - Prone to overfitting on large datasets or complex data.                        | - May not provide a transparent, interpretable model.                                        |
| **Numerical Example:**                         | Consider a small dataset of patient symptoms for diagnosing a common illness. A single decision tree can provide a clear set of rules that a medical practitioner can follow for diagnosis. | In a large telecom dataset with hundreds of features, a random forest can combine multiple decision trees to predict customer churn accurately, considering various factors like call duration, contract length, and customer demographics. |

**Scenario:**

- **When to Prefer:**
  - Decision Tree: When interpretability is crucial, and you need a single, understandable tree.
  - Random Forest: When you seek higher predictive accuracy and robustness to outliers or noisy data.

**Example Use Case:**

- **Medical Diagnosis (Decision Tree):** In a scenario where medical practitioners need clear, explainable rules for diagnosing a common illness based on a small dataset of patient symptoms.
- **Customer Churn Prediction (Random Forest):** When predicting customer churn in a telecom company with a large dataset of diverse features, aiming for improved accuracy.

**Description:**

- **Decision Tree:** Provides a clear, understandable decision path, crucial in scenarios where interpretability is more important than marginal gains in accuracy.
- **Random Forest:** Combines multiple decision trees, reducing overfitting and improving generalization performance, making it suitable for complex, high-dimensional data.

**Advantages:**

- **Decision Tree:** Easy to visualize and explain; works well with small to medium-sized datasets.
- **Random Forest:** Reduces overfitting through ensemble learning; captures complex relationships in data.

**Disadvantages:**

- **Decision Tree:** Prone to overfitting on large datasets or complex data.
- **Random Forest:** May not provide a transparent, interpretable model.

**Numerical Example:**

- **Decision Tree (Medical Diagnosis):** Consider a small dataset of patient symptoms for diagnosing a common illness. A single decision tree can provide a clear set of rules that a medical practitioner can follow for diagnosis.
- **Random Forest (Customer Churn Prediction):** In a large telecom dataset with hundreds of features, a random forest can combine multiple decision trees to predict customer churn accurately, considering various factors like call duration, contract length, and customer demographics.

This table provides a comprehensive comparison of when to prefer decision trees over random forests, including scenarios, advantages, disadvantages, and practical examples.

# 12. Why Logistic Regression is called regression?
Ans:

| Aspect                                    | Logistic Regression                                          |
|-------------------------------------------|-------------------------------------------------------------|
| **Name Justification and Explanation**    | Logistic Regression is called "regression" because it models the probability of an event happening, yielding continuous values between 0 and 1. Despite its classification role, it shares mathematical similarities with linear regression. |
| **Mathematical Formulation**               | Logistic Regression employs the logistic function to model the probability of a binary outcome, yielding continuous probability values within the [0, 1] range based on one or more predictor variables. |
| **Example Use Case**                      | Logistic Regression is applied to predict the probability of a student passing an exam, producing a continuous probability score that quantifies the likelihood of passing based on study hours. |
| **Numerical Example**                     | In the Logistic Regression model, the logistic function is used to express the probability: \( P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X)}} \). Here, \( P(Y=1|X) \) signifies the probability of passing the exam given the number of hours studied, generating a continuous probability value. |

This combined table provides a concise explanation of why Logistic Regression is named as such, emphasizing its role in estimating probabilities as continuous values.

# 13. What is Online Machine Learning? How is it different from Offline machine learning? List some of it’s applications?
Ans:

| Aspect                      | Offline Machine Learning                 | Online Machine Learning                    |
|-----------------------------|----------------------------------------|-------------------------------------------|
| **Definition**               | Trains on a static dataset without the ability to adapt to new data; batch processing. | Continuously updates the model with new data as it becomes available; incremental processing. |
| **Learning Process**         | Batch processing trains the model on the entire dataset at once. | Incremental processing updates the model iteratively as new data arrives. |
| **Data Availability**        | Assumes a fixed dataset available in advance. | Adapts to changing data in real-time, suitable for streaming and dynamic environments. |
| **Examples**                 | Decision trees, Random Forests, Linear Regression. | Online learning algorithms, including Online Gradient Descent, Adaptive Learning, and Streaming K-Means. |
| **Advantages**               | Well-suited for static datasets with known characteristics. | Suitable for applications where data changes over time, enabling timely model updates. |
| **Disadvantages**           | Not ideal for dynamic or streaming data; may lead to outdated models. | May require more computational resources and can be sensitive to parameter settings. |
| **Applications**             | Predictive maintenance, sentiment analysis, image recognition. | Fraud detection, recommendation systems, anomaly detection, stock market forecasting, and online ad targeting. |

**Example**: 

*Online Machine Learning Application: Fraud Detection*

- In an online machine learning system for fraud detection, a bank continuously updates its fraud detection model as new transaction data arrives, adapting to emerging fraud patterns and adjusting predictions in real-time.

This table provides a concise overview of Online Machine Learning compared to Offline Machine Learning, including definitions, learning processes, examples, advantages, disadvantages, and applications, with merged sentences for improved readability.

# 14. What is No Free Lunch Theorem?
Ans:

| Aspect                          | No Free Lunch Theorem Description                                                                                                                        |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Definition**                   | The No Free Lunch Theorem (NFL) is a fundamental concept in machine learning, stating that there is no one-size-fits-all algorithm that outperforms all others across all possible datasets. |
| **Explanation**                  | NFL implies that the effectiveness of a machine learning algorithm depends on the specific characteristics and distribution of the data it is applied to.          |
| **Implications**                 | It underscores the importance of selecting the right algorithm for a specific problem and dataset, as there is no universally superior approach.             |
| **Example**                      | For instance, a decision tree algorithm may perform exceptionally well on one dataset but poorly on another, where a neural network excels. The choice of algorithm should be tailored to the problem. |

**Numerical Example:**

Consider two datasets:

1. **Dataset A:** Contains tabular data with clear linear relationships.
2. **Dataset B:** Contains unstructured text data.

According to the No Free Lunch Theorem, there is no single algorithm that will perform best on both Dataset A and Dataset B. For Dataset A, linear regression might work well, while for Dataset B, natural language processing techniques like word embeddings or deep learning might be more effective. This demonstrates the theorem's core idea that the choice of algorithm depends on the specific dataset and problem.

# 15. Imagine you are working with a laptop of 2GB RAM, how would you process a dataset of 10GB?
Ans:

| Aspect                                | Solution                                                                                                                                                                                                                                                                                                  |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Problem**                           | Processing a 10GB dataset with a 2GB RAM laptop presents a significant challenge due to memory limitations.                                                                                                                                                                                                                                                                 |
| **Description**                       | Limited RAM constraints dataset processing.                                                                                                                                                                                                                                                                                                                               |
| **Solution**                          | 1. **Data Chunking:** Divide the dataset into smaller chunks (e.g., 1GB each) that fit into available RAM. 2. **Sequential Processing:** Process one chunk at a time, analyzing, aggregating, or extracting required information. 3. **Intermediate Storage:** Store intermediate results on disk between chunk processing to free up RAM for the next chunk.                                                                                                     |
| **Numerical Example**                 | If the dataset contains records of 100 million rows, you can load and process approximately 10 million rows at a time, analyze them, store the results on disk, and proceed to the next chunk; this process continues until the entire dataset is processed. For example, if you need to calculate the average age from an age column, you would calculate the average for each chunk and then combine these averages to get the final result. |
| **Advantages**                         | - Enables processing of large datasets with limited resources. - Sequential processing ensures that the entire dataset can be processed, even if it doesn't fit entirely into RAM.                                                                                                                                                                                    |
| **Disadvantages**                     | - Slower processing time compared to processing in-memory. - Requires efficient disk I/O operations and storage space for intermediate results.                                                                                                                                                                                                                        |
| **Considerations**                    | - Chunk size should be chosen carefully to balance processing speed and disk space usage. - Use appropriate data structures and algorithms that can handle chunked processing.                                                                                                                                                                                          |


# 16.  What are the main differences between Structured and Unstructured Data?
Ans:

| Aspect                     | Structured Data                            | Unstructured Data                         |
|----------------------------|--------------------------------------------|-------------------------------------------|
| **Definition**              | Data organized into a predefined format,   | Data lacks a predefined structure and is often in the form of text, images, audio, video, or other raw formats.   |
| **Format**                  | Well-defined, with a clear schema.        | No inherent structure or schema; data may be free-form or semi-structured.                             |
| **Examples**                | Customer information in a relational database, stock market data in a CSV file.               | Social media posts, emails, images, audio recordings, sensor data from IoT.                             |
| **Accessibility**           | Easily queried and analyzed using standard SQL or specialized tools.                             | Requires advanced techniques for data extraction, natural language processing, and machine learning.      |
| **Search and Analysis**     | Quick and straightforward searching and analysis; structured queries.                              | Challenging to search and analyze due to unstructured nature; relies on text and image analysis techniques. |
| **Example Use Case**       | Sales data in a retail store, inventory management in logistics.                                    | Social media sentiment analysis, voice recognition for virtual assistants.                                   |

**Numerical Examples**:

*Structured Data:*

- **Example 1:** Customer Information
  - Table: CustomerID | Name | Age | Address
  - Row 1: 1001 | John Smith | 35 | 123 Main St.
  - Row 2: 1002 | Jane Doe | 28 | 456 Elm St.

- **Example 2:** Stock Market Data
  - Table: Date | Ticker | Price | Volume
  - Row 1: 2023-01-01 | AAPL | 150.25 | 2,000,000
  - Row 2: 2023-01-01 | GOOG | 2800.75 | 1,500,000

*Unstructured Data:*

- **Example 1:** Social Media Post
  - Text: "Just had the best vacation ever! #paradise #travel"
  
- **Example 2:** Audio Recording
  - Format: WAV
  - Audio Analysis Required for Content Extraction


# 17. What are the main points of difference between Bagging and Boosting?
Ans:

| Aspect                                | Bagging                            | Boosting                           |
|---------------------------------------|-----------------------------------|-----------------------------------|
| **Description**                       | Ensemble learning technique that combines multiple base models independently. | Ensemble learning technique that combines multiple base models sequentially. |
| **Examples**                          | Random Forest                      | AdaBoost, Gradient Boosting, XGBoost |
| **Base Model Independence**            | Base models trained independently. | Base models are trained sequentially. |
| **Weighted Voting**                   | Equal weight for each base model.  | Base models weighted based on performance. |
| **Error Correction**                  | Reduces variance (overfitting) by averaging predictions. | Focuses on reducing bias (underfitting) by giving more weight to difficult samples. |
| **Example:**                          | Suppose we have a dataset with 100 base models, each with 90% accuracy. Bagging combines these models, and the ensemble achieves 91% accuracy. | Suppose we have a dataset with 100 base models, where AdaBoost sequentially corrects the errors made by previous models, giving higher weight to misclassified instances. |

# 18. What are the assumptions of linear regression?
Ans:

| Assumption                                  | Description and Example                                           |
|--------------------------------------------|------------------------------------------------------------------|
| **Linearity**                               | The relationship between the independent variables (features) and the dependent variable (target) is linear. For instance, assuming linearity in a house price prediction model means that for every additional square footage increase, the house price increases by a fixed amount, say $100. |
| **Independence of Errors**                  | The errors (residuals) of the regression model are independent of each other.                                           |
| **Homoscedasticity**                        | The variance of the errors is constant across all levels of the independent variables. In other words, the spread of points in a scatterplot of residuals against predicted values should be roughly consistent. |
| **Normality of Errors**                     | The errors follow a normal distribution. You can check this by plotting a histogram of the residuals; it should resemble a bell curve. |
| **No or Little Multicollinearity**           | The independent variables are not highly correlated with each other. For example, in a GPA prediction model that considers high school GPA, SAT score, and extracurricular activities, if the high school GPA and SAT score are highly correlated, it can lead to multicollinearity issues, making it challenging to determine each variable's individual impact on college GPA. |
| **No Endogeneity**                         | There is no endogeneity, meaning that the independent variables are not correlated with the error term. In other words, the model should not suffer from omitted variable bias, where relevant variables are missing from the model. |
| **No Autocorrelation of Errors**            | The errors (residuals) are not correlated with each other over time or across observations. This assumption is particularly important in time series data. |

# 19. How do you measure the accuracy of a Clustering Algorithm?
Ans:

| Aspect                                    | Measurement Method                        | Description and Example                                        |
|-------------------------------------------|------------------------------------------|--------------------------------------------------------------|
| **Accuracy Measurement for Clustering**    | **Silhouette Score** & **Davies-Bouldin Index**| - Silhouette Score measures clustering quality and ranges from -1 to 1. A higher score indicates better clustering. - A Davies-Bouldin Index measures average similarity-to-dissimilarity ratio between clusters. Smaller values suggest more compact, well-separated clusters. | For example, if we obtain a Silhouette Score of 0.65, it indicates well-separated clusters with little overlap. A Davies-Bouldin Index of 1.2 suggests relatively compact, dissimilar clusters. |
|                                           | **Inertia (Within-Cluster Sum of Squares)** | - Inertia measures total distance of data points within clusters from centroids. Lower inertia implies more concentrated clusters. | If the inertia is 1500, it implies data points within clusters are tightly grouped around centroids. |

# 20. What is Matrix Factorization and where is it used in Machine Learning?
Ans:

| Aspect                                | Matrix Factorization                               |
|---------------------------------------|---------------------------------------------------|
| **Description**                       | Matrix factorization is a technique used to decompose a matrix into multiple matrices, often with lower dimensions, revealing latent patterns or features within the data. |
| **Examples**                           | - Singular Value Decomposition (SVD) <br> - Non-Negative Matrix Factorization (NMF) |
| **Use Cases in Machine Learning**     | Matrix factorization is utilized in various machine learning applications, including collaborative filtering for recommender systems, dimensionality reduction, image compression, and topic modeling by decomposing document-term matrices. |
| **Numerical Example**                 | Consider a user-item rating matrix for a movie recommendation system. It's a matrix where rows represent users, columns represent movies, and cells contain user ratings. Matrix factorization can decompose this matrix into two lower-dimensional matrices: one representing users' latent factors and the other representing movies' latent factors. |

# 21. What is an Imbalanced Dataset and how can one deal with this problem?
Ans:

| Aspect                                  | Imbalanced Dataset                                                |
|-----------------------------------------|-------------------------------------------------------------------|
| **Description**                         | An imbalanced dataset is one where the distribution of classes is highly skewed, with one class significantly outnumbering the others, e.g., in a binary classification problem, Class A has 95% of the samples, and Class B has only 5%. |
| **Challenges**                          | Imbalanced datasets pose challenges because machine learning models tend to be biased towards the majority class, leading to poor performance on the minority class. |
| **Dealing with Imbalanced Data**         | Various techniques can address this problem, including: |
|                                         | **Resampling:**   - **Oversampling:** Increase the number of instances in the minority class by duplicating samples or generating synthetic samples, balancing the class distribution.   - **Undersampling:** Decrease the number of instances in the majority class by randomly removing samples. |
|                                         | **Data-Level Methods:**   - **SMOTE (Synthetic Minority Over-sampling Technique):** Generates synthetic samples for the minority class by interpolating between existing samples. |
|                                         | **Algorithmic Techniques:** Use algorithms that handle imbalanced data well, such as Random Forest, Gradient Boosting, or ensemble methods. |
|                                         | **Anomaly Detection:** Treat the minority class as an anomaly detection problem, focusing on detecting rare events. |
|                                         | **Cost-Sensitive Learning:** Assign different misclassification costs to different classes to penalize errors on the minority class. |

**Example:**

Consider a fraud detection scenario where you aim to identify fraudulent credit card transactions. In this case:

- The majority class includes legitimate transactions (95% of data), while the minority class includes fraudulent transactions (5% of data).

To deal with this imbalanced dataset:

- You can apply oversampling to generate more synthetic fraudulent transactions, making the classes more balanced.
- Use an algorithm like Random Forest, which can handle imbalanced data well.
- Implement cost-sensitive learning by assigning higher misclassification costs to fraudulent transactions to increase their importance during model training.

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

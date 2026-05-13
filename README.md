# Credit-Card-Fraud-Detection
Detecting fraudulent transactions is a classic but critical challenge in fintech. For this project, I focused on building a robust pipeline that balances precision and recall to protect users without flagging legitimate purchases.

🔍 The Technical Deep-Dive:

Data Engineering: Performed extensive data cleaning and preprocessing, including handling missing values, one-hot encoding for categorical features, and feature scaling using StandardScaler.

Imbalance Management: Addressed highly imbalanced class distributions (Fraud vs. Non-Fraud) using class_weight strategies and evaluating models via PR-AUC and F1-Scores.

Advanced Modeling: Leveraged Ensemble Learning to boost performance. I implemented and compared:

Random Forest & HistGradientBoosting

Voting Classifiers (Soft Voting)

Stacking Classifiers (using Logistic Regression as a meta-learner)

Data Visualization: Created insightful visuals, including Feature Importance plots to identify key fraud indicators and Heatmap-based Confusion Matrices to track model accuracy.

Deployment: Developed a web-based GUI using Flask, allowing users to interact with the model in real-time.

🛠️ Tech Stack: Python | Pandas | Scikit-learn | Matplotlib | Seaborn | Flask

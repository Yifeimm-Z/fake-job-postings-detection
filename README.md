# Detecting Fraudulent Job Postings: A Data Analysis Approach
A data-driven approach to identifying fake job ads using NLP and machine learning.

## 📖 Project Overview
This project aims to identify and analyze the key characteristics that distinguish fraudulent job postings from legitimate ones. With the rise of online job scams, leveraging data science for early detection can protect job seekers from financial and personal harm. This work demonstrates a complete data analysis pipeline, from data cleaning and exploratory analysis to natural language processing (NLP) and predictive modeling.

**Key Skills Demonstrated:** `Python` `Pandas` `Data Cleaning` `EDA` `Data Visualization` `NLP (NLTK/VADER)` `Machine Learning` `Scikit-learn` `Logistic Regression` `Handling Class Imbalance`

## 🎯 Business Objective
To build an interpretable model that can effectively flag potentially fraudulent job postings based on structured data and text content, providing actionable insights for job platform moderators and job seekers.

## 📁 Dataset
- **Source:** [Real or Fake Job Posting Prediction on Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **License:** `CC0: Public Domain`
- **Size:** ~17,880 job postings with 18 features.
- **Features:** Mix of textual (e.g., `description`, `requirements`), categorical (e.g., `employment_type`, `required_experience`), and binary (e.g., `telecommuting`, `fraudulent`) variables.

## 🔬 Methodology & Analysis
The project follows a standard CRISP-DM workflow:

1.  **Data Preparation & Cleaning**
    - Handled missing values and created indicator features for missing text fields
    - Engineered new features such as text length of descriptions
    - Applied Tidy Data principles for analysis-ready formatting

2.  **Exploratory Data Analysis (EDA)**
    - Uncovered stark patterns: **Remote job postings had a 8.3% fraud rate**, nearly double that of on-site roles (4.7%)
    - Identified that **fraudulent postings often have missing or significantly shorter company profiles and job descriptions**
    - Visualized relationships between categorical features and the target variable

3.  **Natural Language Processing (NLP)**
    - Performed **sentiment analysis** using NLTK's VADER to quantify linguistic differences
    - Conducted **word frequency analysis** to identify distinguishing keywords

4.  **Predictive Modeling**
    - Built a **Logistic Regression classifier** using Scikit-learn
    - Addressed critical **class imbalance** by using `class_weight='balanced'`
    - **Evaluated model performance** with confusion matrix and classification report

## 📊 Key Results & Findings
- The balanced Logistic Regression model achieved:
  - **Accuracy: 0.83**
  - **Recall (Fraudulent class): 0.65** (significant improvement from 0.0 without balancing)
  - **Precision (Fraudulent class): 0.16**
- **Top predictors** of fraudulent postings included missing company profiles, entry-level positions, and telecommuting offers

## 🚀 How to Run This Project
1.  **Clone the repository**
    ```bash
    git clone https://github.com/Yifeimm-Z/fake-job-postings-detection.git
    ```
2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Launch Jupyter Notebook**
    ```bash
    jupyter notebook
    ```
4.  Open and run `fake_job_analysis.ipynb`

## 💡 Future Improvements
- Experiment with advanced NLP techniques (TF-IDF, Word2Vec) for text feature extraction
- Try tree-based models (Random Forest, XGBoost) to capture non-linear relationships
- Deploy as a web application using Flask/Streamlit for demonstration

## 📄 Report
The complete final report is available in the `reports/` directory.

---
*This project was completed as the final assignment for INF1340: Programming for Data Science.*

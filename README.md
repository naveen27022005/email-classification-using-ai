# AI_Based_Email_Classifier


# Project Overview

This project aims to build an AI-powered email classification system for enterprises. Customer service teams receive large volumes of emails, including complaints, requests, feedback, and spam. Manually sorting them is time-consuming and inefficient.
This system automatically classifies emails and predicts urgency levels (high, medium, low) using NLP and machine learning techniques.

# Milestone 1: Data Collection & Preprocessing

## 01. Datasets Used

**A. Enron Email Dataset (Primary Dataset)**

* Real corporate emails (~517,401 raw emails)

* Includes business conversations suitable for multi-class classification

* After cleaning: 510,504 valid emails

**B. SpamAssassin Dataset (Secondary Dataset)**

* Spam emails (501)

* Ham emails (2,801)

* Used for improving spam detection capability

## 02. Preprocessing Steps Completed

* Removed email headers such as: _Message-ID, From, To, Subject, MIME metadata_

* Extracted only the email body (content after first blank line)

* Cleaned noisy text, removed blank lines, trimmed spacing

* Converted all emails to a uniform format using a cleaning function

* Filtered out emails with < 20 characters

* Saved cleaned dataset as enron_clean.csv

* Final cleaned dataset size: 510,504 emails

## 03. Automatic Category Labeling

Using rule-based keyword matching, the following categories were assigned:

* *request* – asking for actions, updates, documents

* *complaint* – problems, issues, errors

* *feedback* – appreciation, suggestions

* *spam* – from SpamAssassin

* *ham* – legitimate non-spam emails

* *other* – does not fit above categories

### Final category distribution:

* *request*	287,686
* *other*	    144,013
* *feedback*	60,867
* *complaint*	17,938
* *ham*	    2,801
* *spam*	    501

## 04. Automatic Urgency Labeling

Based on urgency-related keywords (urgent, asap, immediately, critical, etc.):

* *high* – urgent tasks, system failures, critical issues

* *medium* – needs action but not immediately

* *low* – general communication, greetings, non-actionable

### Final urgency distribution:

* *medium*	245,930
* *low*	    208,814
* *high*	59,062

## 05. Merged Dataset

The Enron and SpamAssassin datasets were merged into a unified dataset with the columns:

    clean_text

    auto_category

    auto_urgency

* It is saved as AI_Based_Email_Classifier/Cleaned data/final_email_dataset.csv

* Total combined emails: 513,806

## 06. Exploratory Data Analysis (EDA) and saving final cleaned and clipped dataset:

### i. Email Length Distribution (Before and After Clipping):

Email lengths were analyzed to detect anomalies.

#### Before Clipping:

* A few emails had lengths exceeding 1–2 million characters, caused by merged threads, HTML noise, or logs.

* These extreme outliers distorted the distribution and required cleaning.

#### After Clipping (<10,000 characters):

* The distribution becomes realistic and right-skewed.

* Most emails have lengths between 200–1500 characters, matching typical enterprise communications.

* Removing outliers improved data quality and model stability.

* This clipped dataset is saved and going to be used in second milestone.

### ii. Email Category Distribution:

This visualization shows how emails are distributed across different categories such as request, feedback, complaint, other, ham, and spam.

#### Key Insights:

* Request emails form the largest proportion of the dataset.

* Other and feedback categories follow next.

* Complaint, ham, and spam categories appear less frequently.

* This class imbalance is expected in enterprise email systems and will be addressed during model training.

### iii. Urgency Level Distribution

Emails were automatically assigned urgency scores: high, medium, and low.

#### Key Insights:

* Most emails fall under medium urgency (~48%).

* Low urgency messages (~41%) include general inquiries and feedback.

* Only 10–12% of emails are high urgency, which aligns with real-world support scenarios where truly urgent emails are fewer.

* This distribution helps prioritize responses for enterprise workflows.

### iv. Category vs. Urgency Heatmap:

This heatmap provides a deeper understanding of how urgency levels vary across different email categories.

#### Key Insights:

* Requests are mostly medium urgency, indicating they need timely action but are not emergencies.

* Feedback and other emails are primarily low urgency.

* Complaints show a mix of low and high urgency—depending on issue severity.

* Spam correctly shows almost no urgency.

* The relationship patterns validate the correctness of the labeling logic.

### v. Wordcloud for Complaint Emails:

A wordcloud was generated to visualize the most frequent terms in complaint emails.

**Key Insights:**

* Dominant words include: error, database, issue, attempting, initialize, operation, engine, and unknown.

* This shows that most "complaint" emails relate to technical or system failures rather than customer-facing complaints.

* The dataset reflects internal company operations (e.g., Enron tech issues).

### Overall Observations:

* The dataset displays natural class imbalance typical of enterprise systems.

* Urgency levels align logically with email categories.

* Email lengths required preprocessing to remove corrupted entries.

* Text patterns in complaints reveal a strong presence of internal IT problems.

* The cleaned dataset is high quality, consistent, and ready for model development in Milestone-2.

---

# Milestone 2: Email Categorization Engine

## Objective
The objective of Milestone 2 is to design and implement an NLP-based email categorization engine that automatically classifies enterprise emails into predefined categories such as **Complaint**, **Request**, **Feedback**, **Spam**, and **Other**. This milestone focuses on model development, training, and evaluation using machine learning and transformer-based approaches.


## Approach

To achieve robust email classification, multiple Natural Language Processing (NLP) techniques and models were explored.

### 1. Text Preprocessing
* This part has been already dealt in Milestone - 1

### 2. Baseline Machine Learning Models
The following baseline classifiers were implemented using Scikit-learn:

- **Logistic Regression**
- **Multinomial Naive Bayes**

These models were trained using TF-IDF vectorized email text and served as a performance benchmark.

### 3. Transformer-Based Model
To improve semantic understanding and contextual accuracy, a transformer-based model was fine-tuned:

- **DistilBERT**

Key steps:
- Tokenization using Hugging Face Transformers
- Fine-tuning on labeled email categories
- Training with class imbalance handling
- Evaluation using standard classification metrics

## Model Evaluation

All models were evaluated using the following metrics:

- **Precision**
- **Recall**
- **F1-Score**
- **Accuracy**

Evaluation was performed on a held-out test dataset to ensure generalization.

### Observations
- Baseline models provided strong performance with fast training times.
- The transformer-based model demonstrated improved contextual understanding, especially for complex and ambiguous email content.
- Class imbalance was handled using weighted loss functions and balanced training strategies.

## Results Summary
- Successful multi-class classification of enterprise emails
- Improved categorization accuracy using transformer-based modeling
- Reliable baseline performance using classical ML models

Detailed evaluation metrics and classification reports are available in the respective notebooks.

---

# Milestone 3: Urgency Detection & Scoring

## Objective
The objective of Milestone 3 is to design and implement an **Urgency Detection Module** that automatically assigns priority levels (**High**, **Medium**, **Low**) to incoming enterprise emails. This ensures that critical issues are identified early and addressed promptly, improving operational efficiency and customer satisfaction.



## Approach

To accurately determine email urgency, a **hybrid approach** combining rule-based heuristics and machine learning techniques was adopted.



### 1. Feature Identification
Urgency-related signals were identified from email content, including:
- Presence of keywords such as *urgent*, *asap*, *immediately*, *not working*, *critical*
- Email length and tone
- Contextual cues indicating deadlines or service outages



### 2. Rule-Based Urgency Detection
A lightweight rule-based system was implemented to quickly flag high-priority emails:
- **High Urgency**: Emails containing strong urgency keywords or failure indicators
- **Medium Urgency**: Emails requesting action without immediate pressure
- **Low Urgency**: Informational or general feedback emails

This approach ensures fast and interpretable urgency classification.



### 3. Machine Learning-Based Urgency Classification
In parallel, a supervised machine learning model was developed:
- TF-IDF vectorization of email text
- Multi-class classification for urgency levels (High / Medium / Low)
- Handling class imbalance using weighted loss strategies

The ML model complements the rule-based system by capturing subtle contextual urgency patterns.



### 4. Hybrid Decision Strategy
The final urgency score is determined by combining:
- Rule-based urgency signals
- Machine learning model predictions

This hybrid strategy improves robustness and reduces misclassification of critical emails.



## Model Evaluation

Urgency detection performance was evaluated using:
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

Evaluation was performed on a held-out test dataset to ensure unbiased assessment.

### Observations
- Rule-based detection effectively captures explicit urgency indicators
- ML-based classification improves accuracy for nuanced cases
- The combined approach provides balanced and reliable urgency predictions



## Results Summary
- Successful classification of emails into **High**, **Medium**, and **Low** urgency levels
- Improved prioritization of time-sensitive emails
- Reliable integration with the email categorization output from Milestone 2

Visualizations and evaluation results are available in the `visualization/` directory.

---

## Note on Model Files

Due to GitHub file size limitations, trained model artifacts (such as `.safetensors`, `.pkl`, and other large binaries) are **not included** in this repository.

All models can be **re-generated** by executing the provided training notebooks in the `Code/` directory.

This follows standard industry practices for machine learning projects and ensures repository portability.

---

# Author

Naveen E, Infosys Springborad Intern

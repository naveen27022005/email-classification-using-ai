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

## 05. Final Merged Dataset

The Enron and SpamAssassin datasets were merged into a unified dataset with the columns:

    clean_text

    auto_category

    auto_urgency

* It is saved as AI_Based_Email_Classifier/Cleaned data/final_email_dataset.csv

* Total combined emails: 513,806

* This file will be used for model training in Milestone 2.

# Author

Naveen E, Infosys Springborad Intern

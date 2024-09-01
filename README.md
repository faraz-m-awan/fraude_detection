# README.md

# Poetry Project: Fraud Detection Model (Featurespace Take Home Test)
**Candidate Name:** Faraz M. Awan
**Position:** Senior Data Scientist


This project focuses on building and training a model to detect fraudulent transactions as a part of Featurespace interview process. The project is divided into two main components: `preprocessing` and `training`. The preprocessing script prepares the data, and the training script builds and evaluates the model.

## Project Structure

```
fraud-detection-project/
│
├── data/
│   ├── transactions_obf.csv
│   ├── labels_obf.csv
│   ├── processed_train_data.csv
│   ├── processed_test_data.csv
│
├── model/
│   └── model.joblib (to be generated after training)
│
├── preprocessing.py
├── train.py
└── pyproject.toml
```

### Files and Folders

- **data/**: This folder contains the original dataset files (`transactions-obf.csv`, `labels_obf.csv`) provided by Featurespace and will also store the processed training and testing datasets.
  
- **model/**: After training, the trained model is saved here as `model.joblib`.

- **preprocessing.py**: This script handles data preprocessing, including feature extraction, handling imbalanced data, and splitting the data into training and testing sets.

- **train.py**: This script loads the processed data, trains the model using a Random Forest Classifier, evaluates its performance, and saves the trained model for deployment if the user is satisfied with the results.

## Prerequisites

This project uses `Poetry` for dependency management and packaging. Make sure you have Poetry installed. If not, you can install it by following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

## Installation

To install the project dependencies, follow these steps:

1. Extract the project folder:


2. Install the dependencies using Poetry:
    ```bash
    poetry install
    ```

3. To activate the virtual environment, use:
    ```bash
    poetry shell
    ```

## Usage

### Preprocessing the Data

To preprocess the data and prepare it for training:

```bash
poetry run python preprocessing.py
```

This script will load the transaction and label data from the `data/` folder, perform feature extraction, handle data imbalance using SMOTE, and save the processed training and testing datasets back into the `data/` folder.

### Training the Model

To train the model and evaluate its performance:

```bash
poetry run python train.py
```

This script will:

1. Load the processed training data.
2. Train a Random Forest Classifier.
3. Evaluate the model on the test data.
4. Display a classification report with precision, recall, and F1-score.

If the user is satisfied with the model's performance, the trained model will be saved in the `model/` folder as `model.pkl`.

### Model Performance

#### Percentage of Fraud Prevention

In addition to traditional classification metrics like precision, recall, and F1-score, another critical metric was calculated: **Percentage of Fraud Prevention**. This metric is particularly significant given the business problem, which involves maximizing the detection of fraudulent transactions within the review capacity of fraud analysts.

**Review Capacity Calculation:**
- The total analyst review capacity was given as 400 transactions per month, which equates to **4,800 transactions per year** (400 transactions/month * 12 months).
- Since the test data represents 30% of the total transactions from one year, the corresponding review capacity for the test set is **1,440 transactions** (4,800 transactions/year * 0.30).

**Calculation Methodology:**
- **Step 1:** For each transaction in the test data, the model generated a probability indicating the likelihood of the transaction being fraudulent.
- **Step 2:** These transactions were then sorted in descending order based on their fraud probability.
- **Step 3:** The top `n` transactions, where `n` equals the analyst review capacity (1,440 transactions in this case), were selected for further analysis.
- **Step 4:** The percentage of total fraudulent transactions detected within this top `n` was calculated.

**Result:** The model was able to identify and prioritize fraudulent transactions effectively, achieving a **95% Fraud Prevention Rate**. This means that 95% of the total fraud in the test set would have been prevented by reviewing the top `n` transactions ranked by the model, significantly reducing potential fraud losses.

This metric demonstrates the model's effectiveness in identifying high-risk transactions, ensuring that the limited review capacity of analysts is used most efficiently to prevent the maximum amount of fraud.

#### Classification Report

Following is the detailed performance of the model on the test data, as calculated using the built-in `classification_report` method from scikit-learn:

          precision    recall  f1-score   support

       0       1.00      1.00      1.00     35324
       1       0.61      0.67      0.64       263

accuracy                           0.99     35587

**Difference from Percentage of Fraud Prevention:**
- **Scope:** The `classification_report` metrics (precision, recall, F1-score) evaluate the model's performance across the entire test dataset, providing an overall view of the model's ability to classify transactions as fraudulent or non-fraudulent.
- **Application:** The **Percentage of Fraud Prevention** metric specifically evaluates the effectiveness of the model in a practical scenario where only a limited number of transactions can be reviewed. It focuses on maximizing the detection of fraud within the top-ranked transactions that the analysts will actually review, rather than assessing the model's performance across the entire test set.

In summary, while the classification report metrics provide a general measure of model performance, the **Percentage of Fraud Prevention** metric directly assesses the real-world impact of the model in preventing fraud under practical constraints.
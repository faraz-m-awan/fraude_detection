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

After training and testing, the model's performance on the test data is summarized as follows:

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| **Class 0** (Non-Fraudulent) | 1.00      | 1.00   | 1.00     | 35324   |
| **Class 1** (Fraudulent)     | 0.61      | 0.67   | 0.64     | 263     |
| **Accuracy**                 |           |        | 0.99     | 35587   |
| **Macro Avg**                | 0.81      | 0.83   | 0.82     | 35587   |
| **Weighted Avg**             | 0.99      | 0.99   | 0.99     | 35587   |


- Class 0: Represents non-fraudulent transactions with a precision, recall, and F1-score of 1.00.
- Class 1: Represents fraudulent transactions with a precision of 0.61, recall of 0.67, and F1-score of 0.64.
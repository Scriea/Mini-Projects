# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset used in this project is sourced from [Kaggle&#39;s Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Introduction

Credit card fraud is a significant issue in the financial sector. This project aims to build a machine learning model that can accurately identify fraudulent transactions. The model is trained and evaluated on a dataset of anonymized credit card transactions, where the goal is to predict the `Class` label (0 for non-fraudulent, 1 for fraudulent).

## Dataset

The dataset contains transactions made by European cardholders in September 2013. It consists of 284,807 transactions, out of which 492 are fraudulent. The dataset is highly imbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.

The dataset can be downloaded from Kaggle using the following [link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Features

The dataset has the following features:

* `V1, V2, ..., V28`: Result of a PCA transformation. These are the principal components obtained with PCA.
* `Time`: The time elapsed between this transaction and the first transaction in the dataset (in seconds).
* `Amount`: The transaction amount.
* `Class`: The label for the transaction (1 for fraud and 0 for non-fraud).

## Installation

To get started with the project, clone the repository and install the necessary dependencies:

<pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4 text-left undefined" dir="ltr"><code class="!whitespace-pre hljs language-bash">git clone https://github.com/screa/Mini-Projects.git
cd Mini-Projects/Credit_Card_Fraud_Detection
pip install -r requirements.txt
</code></div></div></pre>

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data` directory.
2. Run the Jupyter notebook to preprocess the data, train the model, and evaluate the results:

<pre><div class="dark bg-gray-950 rounded-md border-[0.5px] border-token-border-medium"><div class="flex items-center relative text-token-text-secondary bg-token-main-surface-secondary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><div class="flex items-center"><span class="" data-state="closed"><button class="flex gap-1 items-center"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24" class="icon-sm"><path fill="currentColor" fill-rule="evenodd" d="M7 5a3 3 0 0 1 3-3h9a3 3 0 0 1 3 3v9a3 3 0 0 1-3 3h-2v2a3 3 0 0 1-3 3H5a3 3 0 0 1-3-3v-9a3 3 0 0 1 3-3h2zm2 2h5a3 3 0 0 1 3 3v5h2a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1zM5 9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1z" clip-rule="evenodd"></path></svg>Copy code</button></span></div></div><div class="overflow-y-auto p-4 text-left undefined" dir="ltr"><code class="!whitespace-pre hljs language-bash">jupyter notebook Credit_Card_Fraud_Detection.ipynb
</code></div></div></pre>

## Model

The project explores various machine learning algorithms including:

* Logistic Regression
* Decision Trees
* Random Forest
* Gradient Boosting
* Neural Networks

The models are evaluated based on precision, recall, F1-score, and AUC-ROC.

## Results

The best-performing model achieved the following results:

* **Precision** : 0.98
* **Recall** : 0.84
* **F1-Score** : 0.90
* **AUC-ROC** : 0.97

These metrics indicate that the model is effective at identifying fraudulent transactions, balancing the trade-off between false positives and false negatives.

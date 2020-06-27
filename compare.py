import pandas as pd
from sklearn.metrics import accuracy_score


train_true = pd.read_csv("true25000.tsv", delimiter="\t")
train_oracle = pd.read_csv("oracle25000.tsv", delimiter="\t")

# true_reviews = train_true["reviews"]
true_labels = train_true["true_labels"]
# oracle_reviews = train_oracle["reviews"]
oracle_labels = train_oracle["oracle_labels"]

print(accuracy_score(true_labels, oracle_labels))

# print(list(true_reviews) == list(oracle_reviews))

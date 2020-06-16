import pandas as pd
from sklearn.metrics import accuracy_score


train_true = pd.read_csv("true.tsv", delimiter="\t")
train_oracle = pd.read_csv("oracle25000.tsv", delimiter="\t")

true_labels = train_true["true_labels"]
oracle_labels = train_oracle["oracle_labels"]

print(accuracy_score(true_labels, oracle_labels))

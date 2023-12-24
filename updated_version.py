import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances


dataset_path = 'E:/auniverstiy/ML/Project/insurance - Copy.csv'
df = pd.read_csv(dataset_path)
X = df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]  # Features
Y = df['charges']  # Target


# displaying data information before scaling
print(df.describe())


def train_test_split(X, Y, test_size=0.25, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    total_samples = len(X)
    test_samples = int(test_size * total_samples)

    test_indices = random.sample(range(total_samples), test_samples)
    train_indices = [i for i in range(total_samples) if i not in test_indices]

    x_train = X.iloc[train_indices]
    x_test = X.iloc[test_indices]
    y_train = Y.iloc[train_indices]
    y_test = Y.iloc[test_indices]
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.253, random_state=42)


# Scaling features that needs to be scaled because its large range
scaled_features = ['age', 'bmi']
for feature in scaled_features:
    min_age = df[feature].min()
    max_age = df[feature].max()
    avg_age = df[feature].mean()
    df[feature] = (df[feature] - avg_age) / max_age - min_age


print("====================After Scaling========================")

# displaying data information after scaling
print(df.describe())


# KNN Module
def KNN_Module(k, x_train, x_test, y_train, y_test):
    # One-hot encode categorical columns, this function convert string data to integer data to easier deal with it
    x_train = pd.get_dummies(x_train, columns=['sex', 'smoker', 'region'])
    x_test = pd.get_dummies(x_test, columns=['sex', 'smoker', 'region'])
    all_distances = []
    for i in range(len(x_test)):
        distances = []
        for j in range(len(x_train)):
            # Reshape data before computing euclidean distance
            distance = euclidean_distances(x_test.iloc[i].values.reshape(1, -1), x_train.iloc[j].values.reshape(1, -1))
            distances.append(distance[0][0])  # Accessing the distance value from the resulting 2D array
        
        # sort data to get the nearest neighbors
        distances = np.argsort(distances)[:k]
        distances = np.mean(y_train.iloc[distances])
        all_distances.append(distances)
    return all_distances

print("====================Test Data========================")
print(y_test)
print("====================KNN Module========================")
print(KNN_Module(4,x_train, x_test, y_train, y_test))

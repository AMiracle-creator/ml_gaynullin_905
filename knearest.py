import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import Normalizer

from collections import Counter

iris = datasets.load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])
iris_df.head()


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


x = iris_df.iloc[:, :-1]
y = iris_df.iloc[:, -1]

x_train = x.sample(frac=0.8, random_state=0)
y_train = y.sample(frac=0.8, random_state=0)
x_test = x.drop(x_train.index)
y_test = y.drop(y_train.index)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print('x train до нормализации')
print(x_train[0:5])
di = {0.0: 'Setosa', 1.0: 'Versicolor', 2.0: 'Virginica'}

before = sns.pairplot(iris_df.replace({'target': di}), hue='target')
before.fig.suptitle('До нормализации', y=1.08)

scaler = Normalizer().fit(x_train)
normalized_x_train = scaler.transform(x_train)
normalized_x_test = scaler.transform(x_test)

print('x train после нормализации')
print(normalized_x_train[0:5])

## After
iris_df_2 = pd.DataFrame(data=np.c_[normalized_x_train, y_train],
                         columns=iris['feature_names'] + ['target'])
di = {0.0: 'Setosa', 1.0: 'Versicolor', 2.0: 'Virginica'}
after = sns.pairplot(iris_df_2.replace({'target': di}), hue='target')
after.fig.suptitle('После нормализации', y=1.08)


def distance_ecu(x_train, x_test_point):
    distances = []
    for row in range(len(x_train)):
        current_train_point = x_train[row]
        current_distance = 0

        for col in range(len(current_train_point)):
            current_distance += (current_train_point[col] - x_test_point[col]) ** 2
        current_distance = np.sqrt(current_distance)
        distances.append(current_distance)

    distances = pd.DataFrame(data=distances, columns=['dist'])
    return distances


def nearest_neighbors(distance_point, K):
    # сортировка расстояний всех точек
    knearests = distance_point.sort_values(by=['dist'], axis=0)

    # беру ближайщих k соседей
    knearests = knearests[:K]
    return knearests


def most_common(knearest, y_train):
    # получаю индексы видов соседей c их повторениями
    common_types = Counter(y_train[knearest.index])
    # определяю самый популярный вид среди соседей
    prediction = common_types.most_common()[0][0]
    return prediction


def KNN(x_train, y_train, x_test, K):
    prediction = []

    for x_test_point in x_test:
        distance_point = distance_ecu(x_train, x_test_point)
        nearest_point = nearest_neighbors(distance_point, K)
        pred_point = most_common(nearest_point, y_train)
        prediction.append(pred_point)

    return prediction


# функция для подсчета точности вычислений
def calculate_accuracy(y_test, y_pred):
    correct_count = 0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            correct_count = correct_count + 1
    accuracy = correct_count / len(y_test)
    return accuracy


accuracies = []
ks = range(1, 30)
for k in ks:
    y_pred = KNN(normalized_x_train, y_train, normalized_x_test, k)
    accuracy = calculate_accuracy(y_test, y_pred)
    accuracies.append(accuracy)
fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of knn")
plt.show()

sepal_length = np.random.uniform(iris_df.min(axis=0)['sepal length (cm)'], iris_df.max(axis=0)['sepal length (cm)'])
sepal_width = np.random.uniform(iris_df.min(axis=0)['sepal width (cm)'], iris_df.max(axis=0)['sepal width (cm)'])
petal_length = np.random.uniform(iris_df.min(axis=0)['petal length (cm)'], iris_df.max(axis=0)['petal length (cm)'])
petal_width = np.random.uniform(iris_df.min(axis=0)['petal length (cm)'], iris_df.max(axis=0)['petal length (cm)'])

testSet = [[sepal_length, sepal_width, petal_length, petal_width]]
testSet = [[5.6, 2.5, 3.9, 1.1]]
test = pd.DataFrame(testSet)
scaler.transform(test)

max_value = max(accuracies)
max_index = accuracies.index(max_value)
y_pred = KNN(normalized_x_train, y_train, testSet, max_index + 1)
y_pred
import pandas as pd
from seaborn import load_dataset
from IPython.display import display
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np


if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/tmoura/machinelearning/master/transfusion.data'
    df = pd.read_csv(url, header=None)

    #Extraindo o y do X
    y = df[0]
    X = df.drop(0, axis=1)

    #normalizando colunas
    for index in range(1, len(X.columns)+1):
        X[index] = MinMaxScaler().fit_transform(np.array(df[index]).reshape(-1,1))

    #Separar os dados de treinamento e testes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) # 80% treino e 20% teste

    # Treinar 2 árvores de decisão com características diferentes
    model_entropy_tree = DecisionTreeClassifier(criterion="entropy")
    model_gini_tree = DecisionTreeClassifier(criterion="gini")

    model_entropy_tree.fit(X_train, y_train)
    model_gini_tree.fit(X_train, y_train)

    #Treinar 2 kNN com características diferentes
    model_brute_knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean', algorithm='brute')
    model_kd_tree_knn = KNeighborsClassifier(n_neighbors=4, metric='minkowski', algorithm='kd_tree')

    model_brute_knn.fit(X_train, y_train)
    model_kd_tree_knn.fit(X_train, y_train)

    #Medindo o desempenho
    result_entropy_tree = model_entropy_tree.predict(X_test)
    result_gini_tree = model_gini_tree.predict(X_test)
    result_brute_knn = model_brute_knn.predict(X_test)
    result_kd_tree_knn = model_kd_tree_knn.predict(X_test)

    #Resultado
    acc_entropy_tree = metrics.accuracy_score(result_entropy_tree, y_test)
    acc_gini_tree = metrics.accuracy_score(result_gini_tree, y_test)
    acc_brute_knn = metrics.accuracy_score(result_brute_knn, y_test)
    acc_kd_tree_knn = metrics.accuracy_score(result_kd_tree_knn, y_test)


    print(f'accDecisionTree1: {round(acc_entropy_tree * 100)}')
    print(f'accDecisionTree2: {round(acc_gini_tree * 100)}')
    print(f'accKNN1: {round(acc_brute_knn * 100)}')
    print(f'accKNN2: {round(acc_kd_tree_knn * 100)}')

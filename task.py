import os
import csv

import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

result = {}
directory_labels = {
    r'C:\Users\abirb\psd\HL': 0,
    r'C:\Users\abirb\psd\DG': 1,
    r'C:\Users\abirb\psd\DL': 2,
}

frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

for directory in [r'C:\Users\abirb\psd\HL', r'C:\Users\abirb\psd\DG', r'C:\Users\abirb\psd\DL']:
    for dir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, dir)):
            person_label = dir.split('_')[-1]
            person_data = {'task{}'.format(i): [] for i in range(1, 10)}
            for file in os.listdir(os.path.join(directory, dir)):
                if file.endswith('.csv'):
                    with open(os.path.join(directory, dir, file), 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        next(reader)
                        for row in reader:
                            frequency_band = row[0]
                            if frequency_band in frequency_bands:
                                task_number = int(file.split('T')[1].split('.')[0])
                                person_data['task{}'.format(task_number)].extend(row[1:])
            result[person_label] = {'data': person_data, 'label': directory_labels[directory]}

for person_label, person_data in result.items():
    print(f"Person name: {person_label}")
    print(f"Person data: {person_data['data']}")
    print(f"Person label (ground truth): {person_data['label']}\n")

# Print the result dictionary
for person, data in result.items():
    print(f"Person: {person}")
    print(f"Label: {data['label']}")
    print("Frequency bands:")
    for task_number, rows in data['data'].items():
        print(f"- {task_number}: {rows}")
    print()

# Extract the labels
labels= np.array([result[person]['label'] for person in result])
print("labels:")
print (labels)

# Extract the 'task1' band data
task1 = []
for person, data in result.items():
    task1.append(data['data']['task1'])
task1 = np.array(task1)
task1 = np.array(task1, dtype=float)
task1_means = np.mean(task1, axis=1)
print("task1 means:")
for i, mean in enumerate(task1_means):
    print(f"Person {i+1}: {mean}")

task1_means = task1_means.reshape(-1, 1)

print("task1:")
print (task1_means)
print("size of task1")
print(task1_means.shape)


print("__________________ task1 data __________________")
# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(task1_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(task1_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))


best_knn = KNeighborsClassifier(**grid_search_knn.best_params_)
best_knn.fit(x_train, y_train)

y_pred=best_knn.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("RF classifier :")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=4, scoring='accuracy', verbose=1)
grid_search_rf.fit(task1_means, labels)

print("Meilleurs paramètres pour Random Forest: {}".format(grid_search_rf.best_params_))

# Train a RF classifier with the best parameters
best_rf = RandomForestClassifier(**grid_search_rf.best_params_)
best_rf.fit(x_train, y_train)

y_pred=best_rf.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("SVM classifier :")

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=4, scoring='accuracy', verbose=1)
grid_search_svm.fit(task1_means, labels)
print("Meilleurs paramètres pour SVM: {}".format(grid_search_svm.best_params_))

# Train a SVM classifier with the best parameters
best_SVM = SVC(**grid_search_svm.best_params_)
best_SVM.fit(x_train, y_train)

y_pred=best_SVM.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


# Grid Search pour AdaBoost
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

adaboost = AdaBoostClassifier()
grid_search_adaboost = GridSearchCV(adaboost, param_grid_adaboost, cv=4, scoring='accuracy', verbose=1)
grid_search_adaboost.fit(task1_means, labels)
print("Meilleurs paramètres pour adaboost: {}".format(grid_search_adaboost.best_params_))

# Train an adaboost classifier with the best parameters
best_ada = AdaBoostClassifier(**grid_search_adaboost.best_params_)
best_ada.fit(x_train, y_train)

y_pred=best_ada.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)



# Extract the 'task2' band data
task2 = []
for person, data in result.items():
    task2.append(data['data']['task2'])
task2 = np.array(task2)
task2 = np.array(task2, dtype=float)
task2_means = np.mean(task2, axis=1)
print("task2 means:")
for i, mean in enumerate(task2_means):
    print(f"Person {i+1}: {mean}")

task2_means = task2_means.reshape(-1, 1)

print("task2:")
print (task2_means)
print("size of task2")
print(task2_means.shape)


print("__________________ task2 data __________________")
# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(task2_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(task2_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))


best_knn = KNeighborsClassifier(**grid_search_knn.best_params_)
best_knn.fit(x_train, y_train)

y_pred=best_knn.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)



print("RF classifier :")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=4, scoring='accuracy', verbose=1)
grid_search_rf.fit(task2_means, labels)

print("Meilleurs paramètres pour Random Forest: {}".format(grid_search_rf.best_params_))

# Train a RF classifier with the best parameters
best_rf = RandomForestClassifier(**grid_search_rf.best_params_)
best_rf.fit(x_train, y_train)

y_pred=best_rf.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("SVM classifier :")

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=4, scoring='accuracy', verbose=1)
grid_search_svm.fit(task2_means, labels)
print("Meilleurs paramètres pour SVM: {}".format(grid_search_svm.best_params_))

# Train a SVM classifier with the best parameters
best_SVM = SVC(**grid_search_svm.best_params_)
best_SVM.fit(x_train, y_train)

y_pred=best_SVM.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


# Grid Search pour AdaBoost
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

adaboost = AdaBoostClassifier()
grid_search_adaboost = GridSearchCV(adaboost, param_grid_adaboost, cv=4, scoring='accuracy', verbose=1)
grid_search_adaboost.fit(task2_means, labels)
print("Meilleurs paramètres pour adaboost: {}".format(grid_search_adaboost.best_params_))

# Train an adaboost classifier with the best parameters
best_ada = AdaBoostClassifier(**grid_search_adaboost.best_params_)
best_ada.fit(x_train, y_train)

y_pred=best_ada.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)



# Extract the 'task3' band data
task3 = []
for person, data in result.items():
    task3.append(data['data']['task3'])
task3 = np.array(task3)
task3 = np.array(task3, dtype=float)
task3_means = np.mean(task3, axis=1)
print("task3 means:")
for i, mean in enumerate(task3_means):
    print(f"Person {i+1}: {mean}")

task3_means = task3_means.reshape(-1, 1)

print("task3:")
print (task3_means)
print("size of task3")
print(task3_means.shape)


print("__________________ task3 data __________________")
# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(task3_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(task3_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))


best_knn = KNeighborsClassifier(**grid_search_knn.best_params_)
best_knn.fit(x_train, y_train)

y_pred=best_knn.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("RF classifier :")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=4, scoring='accuracy', verbose=1)
grid_search_rf.fit(task3_means, labels)

print("Meilleurs paramètres pour Random Forest: {}".format(grid_search_rf.best_params_))

# Train a RF classifier with the best parameters
best_rf = RandomForestClassifier(**grid_search_rf.best_params_)
best_rf.fit(x_train, y_train)

y_pred=best_rf.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("SVM classifier :")

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=4, scoring='accuracy', verbose=1)
grid_search_svm.fit(task3_means, labels)
print("Meilleurs paramètres pour SVM: {}".format(grid_search_svm.best_params_))

# Train a SVM classifier with the best parameters
best_SVM = SVC(**grid_search_svm.best_params_)
best_SVM.fit(x_train, y_train)

y_pred=best_SVM.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


# Grid Search pour AdaBoost
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

adaboost = AdaBoostClassifier()
grid_search_adaboost = GridSearchCV(adaboost, param_grid_adaboost, cv=4, scoring='accuracy', verbose=1)
grid_search_adaboost.fit(task3_means, labels)
print("Meilleurs paramètres pour adaboost: {}".format(grid_search_adaboost.best_params_))

# Train an adaboost classifier with the best parameters
best_ada = AdaBoostClassifier(**grid_search_adaboost.best_params_)
best_ada.fit(x_train, y_train)

y_pred=best_ada.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)



# Extract the 'task4' band data
task4 = []
for person, data in result.items():
    task4.append(data['data']['task4'])
task4 = np.array(task4)
task4= np.array(task4, dtype=float)
task4_means = np.mean(task4, axis=1)
print("task4 means:")
for i, mean in enumerate(task4_means):
    print(f"Person {i+1}: {mean}")

task4_means = task4_means.reshape(-1, 1)

print("task4:")
print (task4_means)
print("size of task4")
print(task4_means.shape)


print("__________________ task4 data __________________")
# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(task4_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(task4_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))


best_knn = KNeighborsClassifier(**grid_search_knn.best_params_)
best_knn.fit(x_train, y_train)

y_pred=best_knn.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("RF classifier :")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=4, scoring='accuracy', verbose=1)
grid_search_rf.fit(task4_means, labels)

print("Meilleurs paramètres pour Random Forest: {}".format(grid_search_rf.best_params_))

# Train a RF classifier with the best parameters
best_rf = RandomForestClassifier(**grid_search_rf.best_params_)
best_rf.fit(x_train, y_train)

y_pred=best_rf.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("SVM classifier :")

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=4, scoring='accuracy', verbose=1)
grid_search_svm.fit(task4_means, labels)
print("Meilleurs paramètres pour SVM: {}".format(grid_search_svm.best_params_))

# Train a SVM classifier with the best parameters
best_SVM = SVC(**grid_search_svm.best_params_)
best_SVM.fit(x_train, y_train)

y_pred=best_SVM.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


# Grid Search pour AdaBoost
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

adaboost = AdaBoostClassifier()
grid_search_adaboost = GridSearchCV(adaboost, param_grid_adaboost, cv=4, scoring='accuracy', verbose=1)
grid_search_adaboost.fit(task4_means, labels)
print("Meilleurs paramètres pour adaboost: {}".format(grid_search_adaboost.best_params_))

# Train an adaboost classifier with the best parameters
best_ada = AdaBoostClassifier(**grid_search_adaboost.best_params_)
best_ada.fit(x_train, y_train)

y_pred=best_ada.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


# Extract the 'task5' band data
task5 = []
for person, data in result.items():
    task5.append(data['data']['task5'])
task5 = np.array(task5)
task5 = np.array(task5, dtype=float)
task5_means = np.mean(task5, axis=1)
print("task5 means:")
for i, mean in enumerate(task5_means):
    print(f"Person {i+1}: {mean}")

task5_means = task5_means.reshape(-1, 1)

print("task5:")
print (task5_means)
print("size of task5")
print(task5_means.shape)


print("__________________ task5 data __________________")
# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(task5_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(task5_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))


best_knn = KNeighborsClassifier(**grid_search_knn.best_params_)
best_knn.fit(x_train, y_train)

y_pred=best_knn.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("RF classifier :")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=4, scoring='accuracy', verbose=1)
grid_search_rf.fit(task5_means, labels)

print("Meilleurs paramètres pour Random Forest: {}".format(grid_search_rf.best_params_))

# Train a RF classifier with the best parameters
best_rf = RandomForestClassifier(**grid_search_rf.best_params_)
best_rf.fit(x_train, y_train)

y_pred=best_rf.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("SVM classifier :")

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=4, scoring='accuracy', verbose=1)
grid_search_svm.fit(task5_means, labels)
print("Meilleurs paramètres pour SVM: {}".format(grid_search_svm.best_params_))

# Train a SVM classifier with the best parameters
best_SVM = SVC(**grid_search_svm.best_params_)
best_SVM.fit(x_train, y_train)

y_pred=best_SVM.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


# Grid Search pour AdaBoost
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

adaboost = AdaBoostClassifier()
grid_search_adaboost = GridSearchCV(adaboost, param_grid_adaboost, cv=4, scoring='accuracy', verbose=1)
grid_search_adaboost.fit(task5_means, labels)
print("Meilleurs paramètres pour adaboost: {}".format(grid_search_adaboost.best_params_))

# Train an adaboost classifier with the best parameters
best_ada = AdaBoostClassifier(**grid_search_adaboost.best_params_)
best_ada.fit(x_train, y_train)

y_pred=best_ada.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)

# Extract the 'task6' band data
task6 = []
for person, data in result.items():
    task6.append(data['data']['task6'])
task6 = np.array(task6)
task6 = np.array(task6, dtype=float)
task6_means = np.mean(task6, axis=1)
print("task6 means:")
for i, mean in enumerate(task6_means):
    print(f"Person {i+1}: {mean}")

task6_means = task6_means.reshape(-1, 1)

print("task6:")
print (task6_means)
print("size of task6")
print(task6_means.shape)


print("__________________ task6 data __________________")
# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(task6_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(task6_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))


best_knn = KNeighborsClassifier(**grid_search_knn.best_params_)
best_knn.fit(x_train, y_train)

y_pred=best_knn.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("RF classifier :")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=4, scoring='accuracy', verbose=1)
grid_search_rf.fit(task6_means, labels)

print("Meilleurs paramètres pour Random Forest: {}".format(grid_search_rf.best_params_))

# Train a RF classifier with the best parameters
best_rf = RandomForestClassifier(**grid_search_rf.best_params_)
best_rf.fit(x_train, y_train)

y_pred=best_rf.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("SVM classifier :")

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=4, scoring='accuracy', verbose=1)
grid_search_svm.fit(task6_means, labels)
print("Meilleurs paramètres pour SVM: {}".format(grid_search_svm.best_params_))

# Train a SVM classifier with the best parameters
best_SVM = SVC(**grid_search_svm.best_params_)
best_SVM.fit(x_train, y_train)

y_pred=best_SVM.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


# Grid Search pour AdaBoost
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

adaboost = AdaBoostClassifier()
grid_search_adaboost = GridSearchCV(adaboost, param_grid_adaboost, cv=4, scoring='accuracy', verbose=1)
grid_search_adaboost.fit(task6_means, labels)
print("Meilleurs paramètres pour adaboost: {}".format(grid_search_adaboost.best_params_))

# Train an adaboost classifier with the best parameters
best_ada = AdaBoostClassifier(**grid_search_adaboost.best_params_)
best_ada.fit(x_train, y_train)

y_pred=best_ada.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


# Extract the 'task7' band data
task7 = []
for person, data in result.items():
    task7.append(data['data']['task7'])
task7 = np.array(task7)
task7 = np.array(task7, dtype=float)
task7_means = np.mean(task7, axis=1)
print("task7 means:")
for i, mean in enumerate(task7_means):
    print(f"Person {i+1}: {mean}")

task7_means = task7_means.reshape(-1, 1)

print("task7:")
print (task7_means)
print("size of task7")
print(task7_means.shape)


print("__________________ task7 data __________________")
# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(task7_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(task7_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))


best_knn = KNeighborsClassifier(**grid_search_knn.best_params_)
best_knn.fit(x_train, y_train)

y_pred=best_knn.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("RF classifier :")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=4, scoring='accuracy', verbose=1)
grid_search_rf.fit(task7_means, labels)

print("Meilleurs paramètres pour Random Forest: {}".format(grid_search_rf.best_params_))

# Train a RF classifier with the best parameters
best_rf = RandomForestClassifier(**grid_search_rf.best_params_)
best_rf.fit(x_train, y_train)

y_pred=best_rf.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("SVM classifier :")

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=4, scoring='accuracy', verbose=1)
grid_search_svm.fit(task7_means, labels)
print("Meilleurs paramètres pour SVM: {}".format(grid_search_svm.best_params_))

# Train a SVM classifier with the best parameters
best_SVM = SVC(**grid_search_svm.best_params_)
best_SVM.fit(x_train, y_train)

y_pred=best_SVM.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


# Grid Search pour AdaBoost
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

adaboost = AdaBoostClassifier()
grid_search_adaboost = GridSearchCV(adaboost, param_grid_adaboost, cv=4, scoring='accuracy', verbose=1)
grid_search_adaboost.fit(task7_means, labels)
print("Meilleurs paramètres pour adaboost: {}".format(grid_search_adaboost.best_params_))

# Train an adaboost classifier with the best parameters
best_ada = AdaBoostClassifier(**grid_search_adaboost.best_params_)
best_ada.fit(x_train, y_train)

y_pred=best_ada.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)



# Extract the 'task8' band data
task8 = []
for person, data in result.items():
    task8.append(data['data']['task8'])
task8 = np.array(task8)
task8 = np.array(task8, dtype=float)
task8_means = np.mean(task8, axis=1)
print("task8 means:")
for i, mean in enumerate(task8_means):
    print(f"Person {i+1}: {mean}")

task8_means = task8_means.reshape(-1, 1)

print("task8:")
print (task8_means)
print("size of task8")
print(task8_means.shape)


print("__________________ task8 data __________________")
# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(task8_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(task8_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))


best_knn = KNeighborsClassifier(**grid_search_knn.best_params_)
best_knn.fit(x_train, y_train)

y_pred=best_knn.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("RF classifier :")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=4, scoring='accuracy', verbose=1)
grid_search_rf.fit(task8_means, labels)

print("Meilleurs paramètres pour Random Forest: {}".format(grid_search_rf.best_params_))

# Train a RF classifier with the best parameters
best_rf = RandomForestClassifier(**grid_search_rf.best_params_)
best_rf.fit(x_train, y_train)

y_pred=best_rf.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


print("SVM classifier :")

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=4, scoring='accuracy', verbose=1)
grid_search_svm.fit(task8_means, labels)
print("Meilleurs paramètres pour SVM: {}".format(grid_search_svm.best_params_))

# Train a SVM classifier with the best parameters
best_SVM = SVC(**grid_search_svm.best_params_)
best_SVM.fit(x_train, y_train)

y_pred=best_SVM.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


# Grid Search pour AdaBoost
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

adaboost = AdaBoostClassifier()
grid_search_adaboost = GridSearchCV(adaboost, param_grid_adaboost, cv=4, scoring='accuracy', verbose=1)
grid_search_adaboost.fit(task8_means, labels)
print("Meilleurs paramètres pour adaboost: {}".format(grid_search_adaboost.best_params_))

# Train an adaboost classifier with the best parameters
best_ada = AdaBoostClassifier(**grid_search_adaboost.best_params_)
best_ada.fit(x_train, y_train)

y_pred=best_ada.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)


# Extract the 'task9' band data
task9 = []
for person, data in result.items():
    task9.append(data['data']['task9'])
task9 = np.array(task9)
task9 = np.array(task9, dtype=float)
task9_means = np.mean(task9, axis=1)
print("task9 means:")
for i, mean in enumerate(task9_means):
    print(f"Person {i+1}: {mean}")

task9_means = task9_means.reshape(-1, 1)

print("task9:")
print (task9_means)
print("size of task9")
print(task9_means.shape)


print("__________________ task9 data __________________")
# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(task9_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(task9_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))


best_knn = KNeighborsClassifier(**grid_search_knn.best_params_)
best_knn.fit(x_train, y_train)

y_pred=best_knn.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)

print("RF classifier :")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=4, scoring='accuracy', verbose=1)
grid_search_rf.fit(task9_means, labels)

print("Meilleurs paramètres pour Random Forest: {}".format(grid_search_rf.best_params_))

# Train a RF classifier with the best parameters
best_rf = RandomForestClassifier(**grid_search_rf.best_params_)
best_rf.fit(x_train, y_train)

y_pred=best_rf.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)

print("SVM classifier :")

param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]
}
svm = SVC()
grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=4, scoring='accuracy', verbose=1)
grid_search_svm.fit(task9_means, labels)
print("Meilleurs paramètres pour SVM: {}".format(grid_search_svm.best_params_))

# Train a SVM classifier with the best parameters
best_SVM = SVC(**grid_search_svm.best_params_)
best_SVM.fit(x_train, y_train)

y_pred=best_SVM.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)

# Grid Search pour AdaBoost
param_grid_adaboost = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
}

adaboost = AdaBoostClassifier()
grid_search_adaboost = GridSearchCV(adaboost, param_grid_adaboost, cv=4, scoring='accuracy', verbose=1)
grid_search_adaboost.fit(task9_means, labels)
print("Meilleurs paramètres pour adaboost: {}".format(grid_search_adaboost.best_params_))

# Train an adaboost classifier with the best parameters
best_ada = AdaBoostClassifier(**grid_search_adaboost.best_params_)
best_ada.fit(x_train, y_train)

y_pred=best_ada.predict(x_test)
Accuracy_score=metrics.accuracy_score(y_test, y_pred)

conf_mat=metrics.confusion_matrix(y_test, y_pred)
print('\n Confusion Matrix : ', conf_mat)
print('Accuracy Score : ', Accuracy_score)
print('Accuracy in Percentage : ',
int(Accuracy_score*100),'%')
print('\n',classification_report(y_pred,y_test))
# Calculate general precision, recall, and F1-score
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')

print('General Precision : ', precision)
print('General Recall : ', recall)
print('General F1-score : ', f1)
import os
import csv
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Define the frequency bands
frequency_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
# Initialize the result dictionary
result = {}

directory_labels = {
    r'C:\Users\abirb\psd\HL': 0,
    r'C:\Users\abirb\psd\DG': 1,
    r'C:\Users\abirb\psd\DL': 2,

}

# Loop through the 4 directories
for directory in [r'C:\Users\abirb\psd\HL', r'C:\Users\abirb\psd\DG', r'C:\Users\abirb\psd\DL']:
    # Loop through the subdirectories in the current directory
    for dir in os.listdir(directory):
        # Check if the directory is a person's folder
        if os.path.isdir(os.path.join(directory, dir)):
            # Get the person's label from the directory name
            person_label = dir.split('_')[-1]
            print(f"Person: {person_label}")
            # Initialize the person's data dictionary
            person_data = {band: [] for band in frequency_bands}
            # Loop through the 9 CSV files in the person's folder
            for file in os.listdir(os.path.join(directory, dir)):
                if file.endswith('.csv'):
                    # Open the CSV file
                    with open(os.path.join(directory, dir, file), 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        # Skip the header row
                        next(reader)
                        # Loop through the rows in the CSV file
                        for row in reader:
                            # Extract the frequency band from the row
                            frequency_band = row[0]
                            # Check if the frequency band is in the list of frequency bands
                            if frequency_band in frequency_bands:
                                # Skip the first column (frequency band name) and add the rest of the row to the person's data dictionary
                                person_data[frequency_band].extend(row[1:])
            # Add the person's data dictionary to the result dictionary with the person's label
            result[person_label] = {
                'data': person_data,
                'label': directory_labels[directory]
            }

# Get the number of key-value pairs in the dictionary
num_items = len(result)
print(f"The dictionary has {num_items} items.")

# Print the result dictionary
for person, data in result.items():
    print(f"Person: {person}")
    print(f"Label: {data['label']}")
    print("Frequency bands:")
    for band, rows in data['data'].items():
        print(f"- {band}: {rows}")
    print()


# Extract the 'delta' band data
delta_data = []
for person, data in result.items():
    delta_data.append(data['data']['delta'])
delta_data = np.array(delta_data)
delta_data = np.array(delta_data, dtype=float)
delta_means = np.mean(delta_data, axis=1)
print("Delta means:")
for i, mean in enumerate(delta_means):
    print(f"Person {i+1}: {mean}")

delta_means = delta_means.reshape(-1, 1)

print("delta data:")
print (delta_means)
print("size of all data")
print(delta_means.shape)



# Extract the 'theta' band data
theta_data = []
for person, data in result.items():
    theta_data.append(data['data']['theta'])

theta_data = np.array(theta_data)
theta_data = np.array(theta_data, dtype=float)
theta_means = np.mean(theta_data, axis=1)
print("theta means:")
for i, mean in enumerate(theta_means):
    print(f"Person {i+1}: {mean}")

theta_means = theta_means.reshape(-1, 1)

# Extract the 'alpha' band data
alpha_data = []
for person, data in result.items():
    alpha_data.append(data['data']['alpha'])

alpha_data = np.array(alpha_data)
alpha_data = np.array(alpha_data, dtype=float)
alpha_means = np.mean(alpha_data, axis=1)
print("alpha means:")
for i, mean in enumerate(alpha_means):
    print(f"Person {i+1}: {mean}")

alpha_means =alpha_means.reshape(-1, 1)

# Extract the 'beta' band data
beta_data = []
for person, data in result.items():
    beta_data.append(data['data']['beta'])

beta_data = np.array(beta_data)
beta_data = np.array(beta_data, dtype=float)
beta_means = np.mean(beta_data, axis=1)
print("beta means:")
for i, mean in enumerate(beta_means):
    print(f"Person {i+1}: {mean}")

beta_means =beta_means.reshape(-1, 1)

# Extract the 'gamma' band data
gamma_data = []
for person, data in result.items():
    gamma_data.append(data['data']['gamma'])


gamma_data = np.array(gamma_data)
gamma_data = np.array(gamma_data, dtype=float)
gamma_means = np.mean(gamma_data, axis=1)
print("gamma means:")
for i, mean in enumerate(gamma_means):
    print(f"Person {i+1}: {mean}")

gamma_means =gamma_means.reshape(-1, 1)

print("gamma data:")
print (gamma_means)
print("size of all data")
print(gamma_data.shape)

print("beta data:")
print (beta_means)
print("size of all data")
print(beta_data.shape)

print("alpha data:")
print (alpha_means)
print("size of all data")
print(alpha_data.shape)

print("theta data:")
print (theta_means)
print("size of all data")
print(theta_data.shape)



# Extract the labels
labels= np.array([result[person]['label'] for person in result])

print("labels:")
print (labels)



print("__________________ delta data __________________")

# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(delta_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(delta_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))

# Train a KNN classifier with the best parameters
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
grid_search_rf.fit(delta_means, labels)

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
grid_search_svm.fit(delta_means, labels)
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
grid_search_adaboost.fit(delta_means, labels)
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




print("__________________ theta data __________________")

# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(theta_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(theta_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))

# Train a KNN classifier with the best parameters
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
grid_search_rf.fit(theta_means, labels)

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
grid_search_svm.fit(theta_means, labels)
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
grid_search_adaboost.fit(theta_means, labels)
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





print("__________________ alpha data __________________")

# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(alpha_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(alpha_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))

# Train a KNN classifier with the best parameters
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
grid_search_rf.fit(alpha_means, labels)

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
grid_search_svm.fit(alpha_means, labels)
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
grid_search_adaboost.fit(alpha_means, labels)
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






print("__________________ beta data __________________")

# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(beta_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(beta_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))

# Train a KNN classifier with the best parameters
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
grid_search_rf.fit(beta_means, labels)

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
grid_search_svm.fit(beta_means, labels)
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
grid_search_adaboost.fit(beta_means, labels)
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








print("__________________ gamma data __________________")

# display(x_train.shape, y_train.shape, x_test.shape,y_test.shape)
x_train, x_test, y_train, y_test =train_test_split(gamma_means, labels,test_size=0.35,random_state=20)

print("KNN classifier :")
# Perform grid search for KNN parameters
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, param_grid_knn, cv=4, scoring='accuracy', verbose=1)
grid_search_knn.fit(gamma_means, labels)

print("Best parameters for k-NN: {}".format(grid_search_knn.best_params_))

# Train a KNN classifier with the best parameters
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
grid_search_rf.fit(gamma_means, labels)

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
grid_search_svm.fit(gamma_means, labels)
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
grid_search_adaboost.fit(gamma_means, labels)
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






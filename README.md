This project is a Python script that performs EEG signal analysis and classification using machine learning algorithms. The script reads and processes EEG data from multiple subjects and frequency bands, and then extracts statistical features (means) from the data. It then trains several classifiers (KNN, Random Forest, SVM, and AdaBoost) on the feature data and evaluates their performance using cross-validation and various metrics (confusion matrix, accuracy, precision, recall, and F1-score). The script also includes a grid search procedure to optimize the hyperparameters of each classifier.

The script can be used as a starting point for EEG signal analysis and classification tasks, and can be customized and extended according to the specific needs of the application. The code is well-documented and includes comments to explain the different steps and functions.

Features:

Reads and processes EEG data from multiple subjects and frequency bands
Extracts statistical features (means) from the data
Trains several classifiers (KNN, Random Forest, SVM, and AdaBoost) on the feature data
Evaluates classifier performance using cross-validation and various metrics
Includes a grid search procedure to optimize classifier hyperparameters
Well-documented and modular code

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix,roc_curve,accuracy_score,mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('/Users/mac/SalehCordovaTest/machine_learning/Diabetes.csv')

#1
data_desc=data.describe()
data_with_two_dec=data_desc.applymap(lambda x: '{:.2f}'.format(x))
for column in data.columns:
	#removing data that doesn't make sense
	if column=='BMI' or column=='AGE' or column=='PGL' or column=='DIA' or column =='DPF':
		data = data[data[column] != 0]
	print('min value for ' + column +': '+ str(data[column].min()))
	print('max value for ' + column +': '+ str(data[column].max()))
print(data_with_two_dec)

#2
print(data['Diabetic'].value_counts())
sns.countplot(x='Diabetic', data=data)
plt.title('Distribution of Diabetic Class')
plt.xlabel('Diabetic')
plt.ylabel('Count')
plt.show()
#3
age_bins = [20, 30, 40, 50, 60, 70, 80, 90]
age_labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']
data['AgeGroup'] = pd.cut(data['AGE'], bins=age_bins, labels=age_labels)
diabetic_counts = data.groupby(data['AgeGroup'])['Diabetic'].value_counts().unstack()
diabetic_counts.plot(kind='bar', stacked=True)
plt.title('Number of Diabetics in AGE Group')
plt.xlabel('AGE GROUP')
plt.ylabel('Count')
plt.show()
#4 + 5
for column in data.columns:
	if column=='AGE' or column=='BMI':
		plt.figure(figsize=(10, 6))
		sns.kdeplot(data[column], fill=True)
		plt.title('Density Plot for: '+column )
		plt.xlabel(column)
		plt.ylabel('Density')
		plt.show()
#6
corr_matrix = data.drop('AgeGroup', axis=1).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Features')
plt.show()
#2.1
LR1_data = data[['Diabetic', 'AGE']]
LR1_feature = LR1_data.drop('Diabetic', axis=1)
LR1 = LR1_data['Diabetic']             
LR1_feature_train, LR1_feature_test, LR1_train, LR1_test = train_test_split(LR1_feature, LR1, test_size=0.20, random_state=119330)
#feature_train is the training data for both age and diabetus
#feature_test is the testing data for both age and diabetus
#LR1_train is the diabetus train data answer
#LR1_test is the diabetus test data answer
LR = LinearRegression()
LR.fit(LR1_feature_train, LR1_train)
LR1_test_predictions = LR.predict(LR1_feature_test)
rounded_LR1=np.round(LR1_test_predictions)
LR1_accuracy = accuracy_score(LR1_test , rounded_LR1)
print('Accuracy using AGE ONLY: '+str(LR1_accuracy))
LR1_mean_squared_error=mean_squared_error(LR1_test , rounded_LR1)
print('Mean Squared error for LR1: '+str(LR1_mean_squared_error))
#2.2
LR2_data=data[[ 'Diabetic', 'PGL' ]]
LR2_feature = LR2_data.drop('Diabetic', axis=1)
LR2 = LR2_data['Diabetic']
LR2_feature_train, LR2_feature_test, LR2_train, LR2_test = train_test_split(LR2_feature, LR2, test_size=0.20, random_state=119330)
LR = LinearRegression()
LR.fit(LR2_feature_train, LR2_train)
LR2_test_predictions = LR.predict(LR2_feature_test)
rounded_LR2=np.round(LR2_test_predictions)
LR2_accuracy = accuracy_score(LR2_test , rounded_LR2)
LR2_mean_squared_error=mean_squared_error(LR2_test , rounded_LR2)
print('Accuracy using PGL ONLY: '+str(LR2_accuracy))
print('Mean Squared error for LR2: '+str(LR2_mean_squared_error))
#2.3
LR3_data=data[['Diabetic','BMI','AGE','PGL']]
LR3_feature = LR3_data.drop('Diabetic', axis=1)
LR3 = LR3_data['Diabetic']
LR3_feature_train, LR3_feature_test, LR3_train, LR3_test = train_test_split(LR3_feature, LR3, test_size=0.20, random_state=119330)
LR = LinearRegression()
LR.fit(LR3_feature_train, LR3_train)
LR3_test_predictions = LR.predict(LR3_feature_test)
rounded_LR3=np.round(LR3_test_predictions)
LR3_accuracy = accuracy_score(LR3_test , rounded_LR3)
print('Accuracy using PGL AND AGE AND BMI ONLY: '+str(LR3_accuracy))
LR3_mean_squared_error=mean_squared_error(LR3_test , rounded_LR3)
print('Mean Squared error for LR3: '+str(LR3_mean_squared_error))


data=data.drop('AgeGroup' , axis=1)
X = data.drop('Diabetic', axis=1)
y = data['Diabetic']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
k_values = [11, 17, 23, 29]
models = []
accuracies = []
roc_auc_scores = []
conf_matrices = []

for k in k_values:
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the data
    knn.fit(X_train, y_train)
    # Predict on test data
    y_pred = knn.predict(X_test)

    # Store the model
    models.append(knn)
    # Calculate and store accuracy
    accuracies.append(accuracy_score(y_test, y_pred))
    # Calculate and store ROC AUC score
    roc_auc_scores.append(roc_auc_score(y_test, y_pred))
    # Calculate and store the confusion matrix
    conf_matrices.append(confusion_matrix(y_test, y_pred))

# Display the results
for i, k in enumerate(k_values):
    print(f"Results for k={k}:")
    print(f"Accuracy: {accuracies[i]}")
   
plt.figure(figsize=(10, 8))

for i, k in enumerate(k_values):
    fpr, tpr, _ = roc_curve(y_test, models[i].predict_proba(X_test)[:,1])
    plt.plot(fpr, tpr, label=f'K={k} (AUC = {roc_auc_scores[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different k Values in kNN')
plt.legend(loc='best')
plt.show()


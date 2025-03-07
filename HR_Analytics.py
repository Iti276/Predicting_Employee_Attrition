import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.svm import SVC


# Load Data
data = pd.read_csv('Assignment1Dataset_v0.1.csv')

# ----------------------- EDA -----------------------

# 1 - Distribution of Age
plt.figure()
sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')

# 2 - Salary Distribution
plt.figure()
sns.boxplot(x=data['Salary (USD)'])
plt.title('Salary Distribution')

# 3 - Gender and Attrition Relationship
plt.figure()
sns.countplot(x='Gender', hue='Attrition (Target)', data=data)
plt.title('Gender vs Attrition')

# 4 - Salary vs. Performance Rating
plt.figure()
sns.scatterplot(x='Salary (USD)', y='Performance Rating', data=data)
plt.title('Salary vs. Performance Rating')
plt.xlabel('Salary (USD)')
plt.ylabel('Performance Rating')
plt.show()

# ------------------- Data Wrangling -------------------

# Check for missing values
missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Missing Values:\n", missing_values)

# Imputation for Numerical Features
imputer = KNNImputer(n_neighbors=3)
numerical_data = data.select_dtypes(include=['float64', 'int64'])
data_imputed_array = imputer.fit_transform(numerical_data)

# Convert back to DataFrame
data_imputed_numerical = pd.DataFrame(data_imputed_array, columns=numerical_data.columns)

# Combine with Categorical Columns
categorical_data = data.select_dtypes(exclude=['float64', 'int64']).copy()

# Handle missing categorical values
categorical_data.fillna('Unknown', inplace=True)

# Concatenate numerical and categorical data
data_imputed = pd.concat([categorical_data, data_imputed_numerical], axis=1)


# ------------------- Encoding Categorical Data -------------------

categorical_columns = data_imputed.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()

for col in categorical_columns:
    data_imputed[col] = label_encoder.fit_transform(data_imputed[col])

# ------------------- Correlation Heatmap -------------------

correlation_matrix = data_imputed.corr()
correlation_matrix = data_imputed.corr()
correlation_matrix.to_csv('coor_file.csv')
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
#-------Feature Engineering--------#
features = ['Marital Status', 'Years of Experience', 'Salary (USD)', 
             'Performance Rating', 'Working Hours', 'Distance from Home']

X=data_imputed[features]
y=data_imputed['Attrition (Target)']

#-------Split Data into Train and Test Sets-----#
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#-------Train Random Forest Model----------#
rf_model=RandomForestClassifier(n_estimators=100, max_depth=4,min_samples_split=15,random_state=42)
rf_model.fit(X_train,y_train)
#-----Evaluate the Random Forest Model------#
y_pred_rf=rf_model.predict(X_test)
print("Random Forest Classifier Report:")
print(classification_report(y_test,y_pred_rf))
#--------------Feature Importance-------#
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))
#-------cross validating----#
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf_model, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean CV Accuracy:", scores.mean())
#---Apply SMOTE---#
# Check class distribution before SMOTE
print("Class distribution before SMOTE:", Counter(y_train))

# Initialize SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Resample the training dataset
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check class distribution after SMOTE
print("Class distribution after SMOTE:", Counter(y_train_resampled))

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Evaluate Model
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classifier Report:")
print(classification_report(y_test, y_pred_rf))



#-----Class WEIGHTS-----#
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, 
                                  class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Model
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classifier Report:")
print(classification_report(y_test, y_pred_rf))

#----Model 2----------#
# Train SVM Model

svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate the SVM Model
y_pred_svm = svm_model.predict(X_test)
print("Support Vector Machine Classifier Report:")
print(classification_report(y_test, y_pred_svm))













import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectFromModel

data_in = pd.read_csv('training_data.csv')
y = pd.read_csv('training_data_targets.csv')

data_in['targets'] = y
"""To remove missing values"""

#Fields with "--" or "not reported" will be replaced with NA in new_df.
new_df = data_in.replace(to_replace=["--", 'not reported'],
                 value=[pd.NA, pd.NA])
# print(new_df.head(212))
# print(data_in.shape)

#Rows/instances containing atleast one field as NA will be dropped

new_df = new_df.dropna(how='any')
# print(new_df.shape)
# # print(data_in.shape)
# print(new_df.head())

y = new_df['targets']
new_df = new_df.drop(columns='targets')


"""To convert age from string to float"""

# Define a regular expression pattern to match the "X years Y days" format
pattern = r'(\d+)\s+years(?:\s+(\d+)\s+days)?'

# Replace the matched pattern with the desired format
new_df['Age_at_diagnosis'] = new_df['Age_at_diagnosis'].str.replace(pattern, lambda x: x.group(1) if x.group(2) is None else f'{int(x.group(1))}.{str(int(x.group(2))//0.365)[0:-2]}', regex=True).astype(float)

# """Changing MUTATED to 1 and NOT_MUTATED to 0"""

new_df = new_df.replace("MUTATED", 1)
new_df = new_df.replace("NOT_MUTATED", 0)

"""Standardization for numerical features and Encoding for Categorical features"""

categorical_features = ['Gender', 'Primary_Diagnosis','Race']#'IDH1','TP53','ATRX','PTEN','EGFR','CIC','MUC16','PIK3CA','NF1','PIK3R1','FUBP1','RB1','NOTCH1','BCOR','CSMD3','SMARCA4','GRIN2A','IDH2','FAT4','PDGFRA']
numeric_features = ['Age_at_diagnosis']

# Create transformers for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

clf1 = LogisticRegression(solver='lbfgs', max_iter=1000)
clf2 = RandomForestClassifier(n_estimators=100)
clf4 = SVC(probability=True)
voting_clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf4)], voting='soft')

pipeline = Pipeline(steps=[('preprocessor', preprocessor),('feature_selector', SelectFromModel(LinearSVC(dual='auto', penalty='l2'))),  ('classifier', clf1)])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(new_df, y, test_size=0.1, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the testing data
accuracy = pipeline.score(X_test, y_test)
print(f'Accuracy on test data: {accuracy}')


# X_train, X_test, y_train, y_test = train_test_split(new_df, y, random_state=0)

# pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
# print(pipe.fit(X_train, y_train).score(X_test, y_test))
###To convert categorical strings to numerical
# enc = OneHotEncoder()
# print(enc.fit(new_df).categories_)

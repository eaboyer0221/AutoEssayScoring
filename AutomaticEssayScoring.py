
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data into a dataframe
df = pd.read_csv("train.csv").fillna('')

# Create the id_map dataframe
id_map = df[['discourse_id', 'essay_id']]

# Drop the 'essay_id' column from the original dataframe
df = df.drop(columns=['essay_id'])

# Transform the 'quality rating' column into numerical values
rating_map = {'ineffective': -1, 'adequate': 0, 'effective': 1}
df['discourse_effectiveness'] = df['discourse_effectiveness'].map(rating_map)

# Split the data into training and test sets
X_text = df['discourse_text']
X_type = df['discourse_type']
y = df['discourse_effectiveness']
X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=0)
X_train_type, X_test_type, _, _ = train_test_split(X_type, y, test_size=0.2, random_state=0)



# Use TfidfVectorizer to convert text data into numerical vectors
vectorizer = TfidfVectorizer()
X_train_text = vectorizer.fit_transform(X_train_text)
X_test_text = vectorizer.transform(X_test_text)

# Concatenate the text and type data
X_train = pd.concat([pd.DataFrame(X_train_text.toarray()), pd.get_dummies(X_train_type)], axis=1).fillna('')
X_test = pd.concat([pd.DataFrame(X_test_text.toarray()), pd.get_dummies(X_test_type)], axis=1).fillna('')
X_train.columns = X_train.columns.astype(str).fillna('')
X_test.columns = X_test.columns.astype(str).fillna('')


# Train a Random Forest Classifier on the training set
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Predict the quality ratings for the test set
test_predictions = clf.predict(X_test)

# Merge the predictions with the id_map dataframe
df_test = pd.DataFrame({'discourse_id': X_test.index, 'discourse_effectiveness': test_predictions})
merged_df = pd.merge(df_test, id_map)

print(merged_df)

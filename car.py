import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Provide the full path to the CSV file
csv_file_path = r'C:\Users\dell\Python\.venv\Car-price-predictor\data.csv'

# Load the dataset
df = pd.read_csv(csv_file_path)

# Data preprocessing
# Handling missing values (if any)
df = df.dropna()

# Clean 'kms_driven' column: remove ' kms' and convert to int
df['kms_driven'] = df['kms_driven'].str.replace(',', '').str.replace(' kms', '').astype(int)

# Clean 'Price' column: remove commas and non-numeric values
df['Price'] = df['Price'].str.replace(',', '')
df = df[df['Price'].apply(lambda x: x.isnumeric())]  # Keep only rows where Price is numeric
df['Price'] = df['Price'].astype(int)

# Convert categorical data to numeric
label_encoder_carname = LabelEncoder()
label_encoder_carcompany = LabelEncoder()
label_encoder_carfuel_type = LabelEncoder()

df['name'] = label_encoder_carname.fit_transform(df['name'])
df['company'] = label_encoder_carcompany.fit_transform(df['company'])
df['fuel_type'] = label_encoder_carfuel_type.fit_transform(df['fuel_type'])

# Features and target variable
X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the prices on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Save the model and label encoders
joblib.dump(model, 'car_price.pkl')
joblib.dump(label_encoder_carname, 'label_encoder_carname.pkl')
joblib.dump(label_encoder_carcompany, 'label_encoder_carcompany.pkl')
joblib.dump(label_encoder_carfuel_type, 'label_encoder_carfuel_type.pkl')
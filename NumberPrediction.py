# Import necessary libraries
import pandas as pd
import numpy as np

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('data/l_numbers.csv', parse_dates=['date'], dayfirst=True)

# Split the 'numbers' column into separate columns and fill missing values with 0
numbers_split = df['numbers'].str.split(',', expand=True).fillna(0).astype(int)
numbers_columns = [f"number_{i}" for i in range(1, 9)]
numbers_split.columns = numbers_columns

# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# Combine the 'date' column with the split 'numbers' columns
df = pd.concat([df[['date']], numbers_split], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[numbers_columns], df['date'], test_size=0.2, random_state=42)

# Create and train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Find the maximum date in the dataset
last_date = df['date'].max()

# Find the next Saturday date
while last_date.weekday() != 5:  # 5 represents Saturday
    last_date += timedelta(days=1)

# Find the next Saturday's date in the dataset or use random predictions if it doesn't exist
next_saturday = last_date + timedelta(days=7)
next_saturday_data = df[df['date'] == next_saturday]

# Generate 10 different sequences of numbers and calculate their probabilities
num_simulations = 10
simulated_numbers = []
probabilities = []

for _ in range(num_simulations):
    unique_numbers = np.random.choice(range(1, 50), size=8, replace=False)
    simulated_numbers.append(unique_numbers)

    # Predict the probability of this sequence using the trained model
    if not next_saturday_data.empty:
        probability = model.predict_proba([unique_numbers])[0][1]
    else:
        # If the date is not in the dataset, assign a random probability
        probability = np.random.uniform(0, 1)

    probabilities.append(probability)

# Print the results
print("Next Saturday's date:", next_saturday.strftime('%d/%m/%Y'))
print("Simulated numbers for next Saturday with probabilities:")
for i, (numbers, prob) in enumerate(zip(simulated_numbers, probabilities), 1):
    print(f"Simulation {i}: {', '.join(map(str, numbers))} (Probability: {prob:.2f})")
#%%

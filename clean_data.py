import pandas as pd

# Load the raw data
df = pd.read_csv('E0.csv')

# Drop unnecessary columns
df = df.drop(columns=['Div', 'Date', 'Referee'], errors='ignore')

# Drop rows with missing values
df = df.dropna()

# Rename key columns
df = df.rename(columns={
    'FTHG': 'HomeGoals',
    'FTAG': 'AwayGoals',
    'FTR': 'Result'
})

# Encode match outcome as a label
df['Label'] = df['Result'].map({'H': 1, 'D': 0, 'A': -1})

# Save cleaned version
df.to_csv('E0_cleaned.csv', index=False)

print("✅ Cleaned data saved as E0_cleaned.csv")

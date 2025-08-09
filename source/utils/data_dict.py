from pathlib import Path
import pandas as pd

PROJECT_PATH = Path(__file__).parent
print(PROJECT_PATH)

## Creating data dictionary in csv (column names, variables, unique values)
df = pd.read_csv(PROJECT_PATH / 'raw_data/2024_base_sampling_20250624.csv', encoding='cp949')

# Step 1: Get first row as dictionary
data_dict = df.iloc[0].to_dict()

# Step 2: Create key-value DataFrame
key_value_df = pd.DataFrame(list(data_dict.items()), columns=['ColNames', 'Var'])

# Step 3: Add column with unique values from original df
key_value_df['UniqueValues'] = key_value_df['ColNames'].apply(lambda col: df[col].unique().tolist()[1:])

# Step 4: Save to CSV
key_value_df.to_csv('data_dict.csv', index=False)

# Optional: Display result
print(key_value_df)
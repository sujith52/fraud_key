import pandas as pd

# Load only the first 200,000 rows
# This saves memory because it doesn't load all 6 million rows
df = pd.read_csv('dataset.csv', nrows=200000)

# Save it to a new file
df.to_csv('subset_200k.csv', index=False)

print("Done! Saved 200,000 rows to subset_200k.csv")
# main.py

# 1️⃣ Import the Dataset class

from phase6 import Dataset

# 2️⃣ Create a Dataset instance

dataset = Dataset("financials.csv")

# 3️⃣ Load the CSV

dataset.load()
print("First 5 rows:")
print(dataset.data.head())
print(dataset.summarize())

# 4️⃣ Converts currency data to numeric

dataset.clean_currency()

# 5️⃣ Check whether there are NaNs
print(dataset.data.isna().sum())

# 6️⃣ Clean data

dataset.clean("custom", custom=0)
print(dataset.data.isna().sum())

# 7️⃣ Summarize the data
statistics, types = dataset.summarize()
print("\nDescriptive Statistics:")
print(statistics.head())

print("\nColumn Types and Non-Null Counts:")
print(types)

# 8️⃣ Now df inside dataset is cleaned and ready
print("\nCleaned Data Sample:")
print(dataset.data.head())



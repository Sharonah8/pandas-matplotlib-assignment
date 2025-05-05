# ✅ Step 1: Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ✅ Step 2: Load the dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# ✅ Step 3: Preview the data
print("First 5 rows of the dataset:")
print(df.head())

# ✅ Step 4: Check data types and missing values
print("\nData types and missing values:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# ✅ Step 5: Basic statistics
print("\n📊 Descriptive statistics:")
print(df.describe())

# ✅ Step 6: Group by species and compute mean
print("\n📈 Average measurements per species:")
grouped_means = df.groupby('species').mean(numeric_only=True)
print(grouped_means)

# ✅ Step 7: Observations
print("\n🔍 Observations:")
print("Setosa has smaller petal lengths and widths compared to Versicolor and Virginica.")
print("Virginica has the highest measurements overall.")

# ✅ Line Chart
plt.figure(figsize=(10, 5))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.plot(subset.index, subset['petal length (cm)'], label=species)

plt.title('Petal Length Over Samples')
plt.xlabel('Sample Index')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='species', y='petal length (cm)', estimator='mean')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("invented.tsv", sep="\t")

# Display the first few rows of the dataset
print("Dataset Preview:")
#print(df.head())
print(df)

# Calculate basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Create a scatter plot to visualize the relationship between total bill and tip
plt.figure(figsize=(8, 6))
plt.scatter(df['#Canais'], df['Vendas (Euros)'], alpha=0.5)
plt.title('Relationship between #canais and vendas')
plt.xlabel('#Canais')
plt.ylabel('Vendas (Euros)')
plt.grid(True)
plt.show()

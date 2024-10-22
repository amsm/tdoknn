import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load the 'tips' dataset from seaborn
tips = sns.load_dataset('tips')

# Display the first few rows of the dataset
print("Dataset Preview:")
#print(tips.head())
print(tips)

# Calculate basic statistics
print("\nBasic Statistics:")
print(tips.describe())

# Create a scatter plot to visualize the relationship between total bill and tip
plt.figure(figsize=(8, 6))
plt.scatter(tips['total_bill'], tips['tip'], alpha=0.5)
plt.title('Relationship between Total Bill and Tip')
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.grid(True)
plt.show()

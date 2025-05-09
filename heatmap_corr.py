import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset (example using sklearn)
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame  # Convert to Pandas DataFrame

# Display first 5 rows
#print(df.head())
#print(df.info())

# Calculate correlation matrix
corr = df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

# Set title
plt.title('Correlation Heatmap of California Housing Dataset')
plt.show()

strong_corr = corr[abs(corr["MedHouseVal"]) > 0.3]
print(strong_corr.index.tolist())
"""
sns.scatterplot(data=df, x="MedInc", y="MedHouseVal", alpha=0.5)
plt.title("Income vs. House Value")

# Show plot
plt.show()
"""
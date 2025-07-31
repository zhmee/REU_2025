import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#### CHANGE DATASET PATH - LOCATED IN pp1 ########################################
df = pd.read_csv('/umbc/rs/cybertrn/reu2025/team2/research/base/pp1/data/patient_combined.csv')

last_column = df.iloc[:, -1]

# Count the occurrences of each class
class_counts = last_column.value_counts()

print("Class Distribution of Data: ")
print(class_counts)

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title('Class Distribution of Last Column')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right') # Rotate labels if many classes
plt.tight_layout()
plt.show()

output_folder = 'viz_classes'
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'patient_combined.png')        ## CHANGE FIGURE NAME ##########################
plt.savefig(output_path)
print(f"\nFigure saved to: {output_path}")

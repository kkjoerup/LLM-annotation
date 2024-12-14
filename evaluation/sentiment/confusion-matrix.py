import pandas as pd
from sklearn.metrics import *
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
annotations = pd.read_csv("")

# Function to map annotations to integer values
def map_sentiment_to_int(annotation):
    if "Negativ" in annotation or "negativ" in annotation:
        return 0
    elif "Neutral" in annotation or "neutral" in annotation:
        return 1
    elif "Positiv" in annotation or "positiv" in annotation:
        return 2
    else:
        return -1

## Mapping
reference_annotations = annotations["label"].map(map_sentiment_to_int)
llm_annotations = annotations["llm_annotation"].map(map_sentiment_to_int)
# Replace NaN values with 'Unannotated'
reference_annotations = reference_annotations.fillna(4)
llm_annotations = llm_annotations.fillna(4)

# Confusion matrix
cm = confusion_matrix(reference_annotations, llm_annotations)

# Plot
plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title("Confusion matrix example: Human and LLM annotations", fontsize=14)
plt.xlabel("LLM annotations", fontsize=12, labelpad=10)
plt.ylabel("Human annotations", fontsize=12, labelpad=10)
plt.xticks(ticks=[0.5, 1.5, 2.5], labels=["Neutral", "Negative", "Positive", "Unannotated"], fontsize=10)
plt.yticks(ticks=[0.5, 1.5, 2.5], labels=["Neutral", "Negative", "Positive", "Unannotated"], fontsize=10)

# Display the plot
plt.tight_layout()
plt.savefig("", dpi=300, bbox_inches="tight")
plt.show()

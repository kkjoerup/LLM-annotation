import pandas as pd
from sklearn.metrics import *
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Load data
annotations = pd.read_csv("")

# Format topics from strings to lists
def annotation_to_list(annotation):
    annotation_list = []
    
    topics = ["Begivenhed", "Bolig", "Erhverv", "Dyr", "Katastrofe", "Kendt", "Konflikt og krig", "Kriminalitet", "Kultur", "Livsstil",
          "Politik", "Samfund", "Sport", "Sundhed", "Teknologi", "Transportmiddel", "Uddannelse", "Underholdning", "Vejr", "Videnskab", "Økonomi"]

    # Regular expression to capture words inside single/double quotes or just words
    annotation_topics = [topic.lower() for topic in re.findall(r'["\']?([a-zA-ZæøåÆØÅ\s]+)["\']?', annotation)]
    
    # Iterate over the topics and check for case-insensitive matches in the extracted topics
    for topic in topics:
        if topic.lower() in annotation_topics:
            annotation_list.append(topic)
    
    return annotation_list

# Transform annotations to lists
annotations["label"] = annotations["label"].apply(annotation_to_list)
annotations["llm_annotation"] = annotations["llm_annotation"].apply(annotation_to_list)

# Add number of labels
annotations["human-no-of-labels"] = annotations["label"].apply(len)
annotations["llm-no-of-labels"] = annotations["llm_annotation"].apply(len)

# Confusin matrix
cm = confusion_matrix(annotations["human-no-of-labels"], annotations["llm-no-of-labels"])

# Plot
plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion matrix example: Human and LLM annotations", fontsize=14)
plt.xlabel("LLM annotations", fontsize=12, labelpad=10)
plt.ylabel("Human annotations", fontsize=12, labelpad=10)

# Display the plot
plt.tight_layout()
plt.savefig("", dpi=300, bbox_inches="tight")
plt.show()
import pandas as pd
from sklearn.metrics import *

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
        return -1  # In case of unexpected values
    
# Evaluation functions
def evaluate_single_label_without_unannotated(reference: pd.Series, llm_annotation: pd.Series):
    # Map strings to integers
    reference = reference.map(map_sentiment_to_int)
    llm_annotation = llm_annotation.map(map_sentiment_to_int)

    # Identify indices where series1 is -1
    unannotated_indices = llm_annotation[llm_annotation == -1].index
    print("Number of unannotated samples:", len(unannotated_indices))
    # Drop the identified indices from both Series
    reference = reference.drop(unannotated_indices)
    llm_annotation = llm_annotation.drop(unannotated_indices)

    # Evaluation metrics
    accuracy = round(accuracy_score(reference, llm_annotation),2)
    macro_precision = round(precision_score(reference, llm_annotation, average = "macro"), 2)
    macro_recall = round(recall_score(reference, llm_annotation, average = "macro"), 2)
    macro_f1 = round(f1_score(reference, llm_annotation, average = "macro"), 2)
    micro_f1 = round(f1_score(reference, llm_annotation, average = "micro"),2)

    return f"""
Evaluation WITHOUT unannotated items:
Accuracy: {accuracy}
Macro-precision: {macro_precision}
Macro-recall: {macro_recall}
Micro-f1: {micro_f1}
Macro-f1: {macro_f1}
"""

def evaluate_single_label_with_unannotated(reference: pd.Series, llm_annotation: pd.Series):
    # Map strings to integers
    reference = reference.map(map_sentiment_to_int)
    llm_annotation = llm_annotation.map(map_sentiment_to_int)

    # Evaluation metrics
    accuracy = round(accuracy_score(reference, llm_annotation),2)
    macro_precision = round(precision_score(reference, llm_annotation, average = "macro"), 2)
    macro_recall = round(recall_score(reference, llm_annotation, average = "macro"), 2)
    macro_f1 = round(f1_score(reference, llm_annotation, average = "macro"), 2)
    micro_f1 = round(f1_score(reference, llm_annotation, average = "micro"),2)

    return f"""
Evaluation WITH unannotated items:
Accuracy: {accuracy}
Macro-precision: {macro_precision}
Macro-recall: {macro_recall}
Micro-f1: {micro_f1}
Macro-f1: {macro_f1}
"""

with_unannotated = evaluate_single_label_with_unannotated(reference=annotations["label"], llm_annotation=annotations["llm_annotation"])
print(with_unannotated)

without_unannotated = evaluate_single_label_without_unannotated(reference=annotations["label"], llm_annotation=annotations["llm_annotation"])
print(without_unannotated)

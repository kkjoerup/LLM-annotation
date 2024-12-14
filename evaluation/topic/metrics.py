import pandas as pd
import re
from sklearn.metrics import hamming_loss
from sklearn.metrics import *
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt

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

# Multi-label accuracy
def multi_label_accuracy(reference: pd.Series, llm_annotation: pd.Series):
    accuracies = []
    for ref_labels, llm_labels in zip(reference, llm_annotation):
        # total set of unique labels present in either the reference or llm_an
        # returns the elements that are present in both sets
        correct_labels = set(ref_labels) & set(llm_labels)
        # total set of unique labels present in either the reference or llm_annotation sets
        total_labels = set(ref_labels) | set(llm_labels)
        accuracy = len(correct_labels) / len(total_labels) if total_labels else 1
        accuracies.append(accuracy)
    return sum(accuracies) / len(accuracies) # average accuracy

# Evaluation functions
def evaluate_multi_label_without_unannotated(reference: pd.Series, llm_annotation: pd.Series):
    reference = reference.apply(annotation_to_list)
    llm_annotation = llm_annotation.apply(annotation_to_list)
    # Number of unannotated cases
    print('Number of unannotated samples:', llm_annotation.apply(lambda x: isinstance(x, list) and len(x) == 0).sum())
    # remove unannotated samples
    # Identify indices where llm_annotation contains empty lists
    empty_list_indices = llm_annotation[llm_annotation.apply(lambda x: x == [])].index
    # Drop the identified indices from both Series
    reference = reference.drop(empty_list_indices)
    llm_annotation = llm_annotation.drop(empty_list_indices)

    # sklearn binarizer
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(reference)
    binarized_references = mlb.transform(reference)
    binarized_llm_annotations = mlb.transform(llm_annotation)
    # Evaluation metrics
    accuracy = round(multi_label_accuracy(reference, llm_annotation), 2)
    precision = round(precision_score(binarized_references, binarized_llm_annotations, average='samples'), 2)
    recall = round(recall_score(binarized_references, binarized_llm_annotations, average='samples'), 2)
    f1score = round(f1_score(binarized_references, binarized_llm_annotations, average='samples'), 2)
    hl = round(hamming_loss(binarized_references, binarized_llm_annotations), 2)
    return f"""
Evaluation WITHOUT unannotated samples:
Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1-score: {f1score}
Hamming-loss: {hl}
"""

# Evaluation function
def evaluate_multi_label_with_unannotated(reference: pd.Series, llm_annotation: pd.Series):
    # LLM annotations to lists of topics
    reference = reference.apply(annotation_to_list)
    llm_annotation = llm_annotation.apply(annotation_to_list)
    # get number of unannotated cases
    print('Number of unannotated samples:', llm_annotation.apply(lambda x: isinstance(x, list) and len(x) == 0).sum())
    # sklearn binarizer
    mlb = MultiLabelBinarizer()
    mlb.fit_transform(reference)
    binarized_references = mlb.transform(reference)
    binarized_llm_annotations = mlb.transform(llm_annotation)
    # Evaluation metrics
    accuracy = round(multi_label_accuracy(reference, llm_annotation), 2)
    precision = round(precision_score(binarized_references, binarized_llm_annotations, average='samples'), 2)
    recall = round(recall_score(binarized_references, binarized_llm_annotations, average='samples'), 2)
    f1score = round(f1_score(binarized_references, binarized_llm_annotations, average='samples'), 2)
    hl = round(hamming_loss(binarized_references, binarized_llm_annotations), 2)
    return f"""
Evaluation WITH unannotated samples:
Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1-score: {f1score}
Hamming-loss: {hl}
"""

with_unannotated = evaluate_multi_label_with_unannotated(reference=annotations["label"], llm_annotation=annotations["llm_annotation"])
print(with_unannotated)

without_unannotated = evaluate_multi_label_without_unannotated(reference=annotations["label"], llm_annotation=annotations["llm_annotation"])
print(without_unannotated)

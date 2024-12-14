import pandas as pd
import re
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.distance import jaccard_distance

# Load data
data = pd.read_csv("")

# Format topics from strings to lists
def annotation_to_list(annotation):
    annotation_list = []
    
    topics = ["Begivenhed", "Bolig", "Erhverv", "Dyr", "Katastrofe", "Kendt", "Konflikt og krig", "Kriminalitet", "Kultur", "Livsstil",
          "Politik", "Samfund", "Sport", "Sundhed", "Teknologi", "Transportmiddel", "Uddannelse", "Underholdning", "Vejr", "Videnskab", "Økonomi"]
    
    # Regular expression to capture words inside single/double quotes or just words
    llm_annotation_topics = [topic.lower() for topic in re.findall(r'["\']?([a-zA-ZæøåÆØÅ\s]+)["\']?', annotation)]
    
    # Iterate over the topics and check for case-insensitive matches in the extracted topics
    for topic in topics:
        if topic.lower() in llm_annotation_topics:  # Case-insensitive match
            annotation_list.append(topic)
    
    return annotation_list

# IAA between two annotators
def iaa_two_annotators(data, remove):
    data["label"] = data["label"].apply(annotation_to_list)
    data["llm_annotation"] = data["llm_annotation"].apply(annotation_to_list)
    
    # Remove rows where "llm_annotation" is empty list
    if remove == True:
        print("len before removing unannoted items:", len(data))
        data = data[data["llm_annotation"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        print("len after removing unannoted items:", len(data))

    annotation_data = data[["id","label","llm_annotation"]]
    # Define annotators
    annotators = ["label","llm_annotation"]
    annotator_names = ["human", "llm"]

    # Format data
    formatted_data = []
    for idx, row in annotation_data.iterrows():
        for i, annotator in enumerate(annotators):
            # Check if the annotation is a valid list and not empty
            if row[annotator] and isinstance(row[annotator], list):
                annotation_set = frozenset(row[annotator])
                formatted_data.append((annotator_names[i], row["id"], annotation_set))

    jaccard_task = nltk.AnnotationTask(distance=jaccard_distance)
    jaccard_task.load_array(formatted_data)

    return "Alpha with two annotators: {:.2f}".format(jaccard_task.alpha())

# IAA between three annotators
def iaa_three_annotators(data, remove):
    data = data[data["second_label"].isna() == False]

    data["label"] = data["label"].apply(annotation_to_list)
    data["llm_annotation"] = data["llm_annotation"].apply(annotation_to_list)
    data["second_label"] = data["second_label"].apply(annotation_to_list)

    # Remove rows where "llm_annotation" is empty list
    if remove == True:
        print("len before removing unannoted items:", len(data))
        data = data[data["llm_annotation"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        print("len after removing unannoted items:", len(data))

    annotation_data = data[["id","label","llm_annotation","second_label"]]

    # Define annotators
    annotators = ["label","llm_annotation","second_label"]
    annotator_names = ["human1", "human2","llm"]

    # Format data
    formatted_data = []
    for idx, row in annotation_data.iterrows():
        for i, annotator in enumerate(annotators):
            # Check if the annotation is a valid list and not empty
            if row[annotator] and isinstance(row[annotator], list):
                annotation_set = frozenset(row[annotator])
                formatted_data.append((annotator_names[i], row["id"], annotation_set))

    jaccard_task = nltk.AnnotationTask(distance=jaccard_distance)
    jaccard_task.load_array(formatted_data)

    return "Alpha with 3 annotators: {:.2f}".format(jaccard_task.alpha())

print("Results including unannotated items")
iaa_two = iaa_two_annotators(data.copy(), remove = False)
print(iaa_two)
iaa_three = iaa_three_annotators(data.copy(), remove = False)
print(iaa_three)
print("------------")
print("Results excluding unannotated items")
iaa_two = iaa_two_annotators(data.copy(), remove = True)
print(iaa_two)
iaa_three = iaa_three_annotators(data.copy(), remove = True)
print(iaa_three)

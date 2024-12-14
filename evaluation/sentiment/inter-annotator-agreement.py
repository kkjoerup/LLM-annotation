import pandas as pd
import krippendorff

# Load data
data = pd.read_csv("")

# Map strings to integers
def map_sentiment_to_int(annotation):
    if "Negativ" in annotation or "negativ" in annotation:
        return 0
    elif "Neutral" in annotation or "neutral" in annotation:
        return 1
    elif "Positiv" in annotation or "positiv" in annotation:
        return 2
    else:
        return None 

# Between two annotators   
def iaa_two_annotators(data, remove):
    data["label"] = data["label"].map(map_sentiment_to_int)
    data["llm_annotation"] = data["llm_annotation"].map(map_sentiment_to_int)
    
    # Remove rows where "llm_annotation" is nan
    if remove == True:
        print("len before removing unannoted items:", len(data))
        data = data[data["llm_annotation"].notna()]
        print("len after removing unannoted items:", len(data))

    annotation_data = data[["label", "llm_annotation"]].T.values.tolist()

    return f"alpha", round(krippendorff.alpha(reliability_data=annotation_data, level_of_measurement="nominal"), 2)

# Between three annotators
def iaa_three_annotators(data, remove):
    data = data[data["second_label"].isna() == False]

    data["label"] = data["label"].map(map_sentiment_to_int)
    data["second_label"] = data["second_label"].map(map_sentiment_to_int)
    data["llm_annotation"] = data["llm_annotation"].map(map_sentiment_to_int)

    #Remove rows where "llm_annotation" is nan
    if remove == True:
        print("len before removing unannoted items:", len(data))
        data = data[data["llm_annotation"].notna()]
        print("len after removing unannoted items:", len(data))

    annotation_data = data[["label", "second_label", "llm_annotation"]].T.values.tolist()

    return f"alpha", round(krippendorff.alpha(reliability_data=annotation_data, level_of_measurement="nominal"), 2)

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

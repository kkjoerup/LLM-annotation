import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

#API key
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

# load evaluation data
data = pd.read_csv("")

# prompt
sentiment_v1_prompt = """
Du er en gennemsnitlig dansk nyhedsforbruger. Du får en overskrift og underoverskrift på en artikel, og skal tildele den en kategori svarende til det sentiment den fremkalder.
Kategorier: ”Positiv”: Fremkalder en overordnet positiv sentiment. ”Negativ”: Fremkalder en overordnet negativ sentiment. ”Neutral”: Fremkalder hverken en positiv eller negativ sentiment
Giv et præcist svar i json: {{sentiment: ”kategori”}}.
"""

# Function to call GPT-4o with zero-shot prompts
def zeroshot_sentiment_annotation(text, prompt):
    try:
        # Make a request to GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system", 
                    "content": f"{prompt}"
                },
                {
                    "role": "user", 
                    "content": f"Artikel: {text} \nArtiklen fremkalder dette sentiment:"
                }
            ],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "topic_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "description": "Sentiment of the article",
                                "type": "string"
                            },
                            "confidence": {
                                "description": "Confidence score for the sentiment from 0.00 to 1.00",
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0
                            }
                        },
                        "additionalProperties": False
                    }
                }
            }
        )

        # Extract response
        sentiment_data = response.choices[0].message.content
        return sentiment_data
    except Exception as e:
        print(f"Error: {e}")
        return None
    

# Apply function and add results to df
data["llm_annotation"] = data["text"].apply(lambda text: zeroshot_sentiment_annotation(text, prompt=sentiment_v1_prompt))

# Save the results to a new CSV file
data.to_csv("", index=False)

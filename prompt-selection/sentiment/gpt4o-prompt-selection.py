import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

#API key
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

# Prompts
sentiment_v0_prompt = """
Du er en gennemsnitlig dansk nyhedsforbruger. Du får en overskrift og underoverskrift på en artikel, og skal tildele den en kategori svarende til det sentiment den fremkalder.
Kategorier: ”Positiv”, ”Negativ”, ”Neutral”.
Giv et præcist svar i json: {{sentiment: ”kategori”}}.
"""
sentiment_v1_prompt = """
Du er en gennemsnitlig dansk nyhedsforbruger. Du får en overskrift og underoverskrift på en artikel, og skal tildele den en kategori svarende til det sentiment den fremkalder.
Kategorier: ”Positiv”: Fremkalder en overordnet positiv sentiment. ”Negativ”: Fremkalder en overordnet negativ sentiment. ”Neutral”: Fremkalder hverken en positiv eller negativ sentiment
Giv et præcist svar i json: {{sentiment: ”kategori”}}.
"""
sentiment_v2_prompt = """
Du er en gennemsnitlig dansk nyhedsforbruger. Du får en overskrift og underoverskrift på en artikel, og skal tildele den en kategori svarende til det sentiment den fremkalder. Sentiment skal vurderes ud fra en gennemsnitlig dansk nyhedsforbrugers perspektiv, og du skal bruge din viden om det danske samfund og undgå personlige holdninger og bias. Tildel artiklen dens mest dominerende sentiment, vær opmærksom på både explicit og implicit sentiment, og brug din generelle viden om følelsesladede udtryk.
Kategorier: ”Positiv”: Fremkalder en overordnet positiv sentiment. Stemninger som optimisme, tilfredshed og selvsikkerhed betragtes som positive. ”Negativ”: Fremkalder en overordnet negativ sentiment. Stemninger som vrede, skuffelse og tristhed betragtes som negative. ”Neutral”: Fremkalder hverken en positiv eller negativ sentiment. Enten ingen sentiment eller tvetydig sentiment.
Giv et præcist svar i json: {{sentiment: ”kategori”}}.
"""
sentiment_v3_prompt = """
Du er en gennemsnitlig dansk nyhedsforbruger. Du får en overskrift og underoverskrift på en artikel, og skal tildele den en kategori svarende til det sentiment den fremkalder. 
Her er nogle generelle principper, du skal følge:
Du skal bruge din viden om det danske samfund og annotere artiklen som du forestiller dig, at den gennemsnitlige dansker ville gøre det.
Undgå at lade dig påvirke af personlige holdninger og bias. Nogle artikler udtrykker holdninger som du måske er uenig med, men det må ikke påvirke vurderingen af artiklens sentiment.
Du skal tildele artiklen dens mest dominerende sentiment, hvis den vurderes at indeholde flere sentiments.
Der findes både explicit og implicit sentiment. Explicit sentiment afspejler ofte nogens indre tilstand (f.eks. tro, holdninger, tanker, følelser osv.). Implicit sentiment afspejler derimod ofte en kendsgerning, der fører til et positivt eller negativt sentiment omkring et emne (f.eks. en person eller begivenhed). Artikler fremkalder ofte implicit sentiment ved at fremhæve gode eller dårlige begivenheder. F.eks. er begivenheder som fødsler, at blive gift eller forfremmet gode begivenheder, mens f.eks. død og sygdom er dårlige begivenheder.
Du skal bruge din generelle viden om følelsesladede udtryk på dansk. Vær opmærksom på, at sentiment kan forekomme ved ord, der normalt ikke betragtes som følelsesladede, f.eks. familie, løn og ansættelse. Nogle begivenheder er også forbundet med sentiment, f.eks. 1. verdenskrig eller Covid-19.
Kategorier: ”Positiv”: Fremkalder en overordnet positiv sentiment. Stemninger som optimisme, tilfredshed og selvsikkerhed betragtes som positive. ”Negativ”: Fremkalder en overordnet negativ sentiment. Stemninger som vrede, skuffelse og tristhed betragtes som negative. ”Neutral”: Fremkalder hverken en positiv eller negativ sentiment. Enten ingen sentiment eller tvetydig sentiment. 
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
                            "additionalProperties": False
                        }
                    }
                }
            }
        )

        # Extract  response
        sentiment = response.choices[0].message.content
        return sentiment
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to call GPT-4o with few-shot prompts
def fewshot_sentiment_annotation(text, prompt, fewshot_dataset):
    # Sample few-shot examples from dataset without current article
    fewshot_dataset = fewshot_dataset[fewshot_dataset['text'] != text]
    fewshot_examples = fewshot_dataset.sample(3).to_dict(orient='records')
    # Create few-shot part of prompt
    fewshot_examples = "\n".join([f"Artikel: {example['text']}. Artiklen fremkalder dette sentiment: {example['label']}." for example in fewshot_examples])

    try:
        # Make a request to GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system", 
                    "content": f"{prompt}. Her er tre eksempler på artikler og deres sentiment: {fewshot_examples}"
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
                            "additionalProperties": False
                        }
                    }
                }
            }
        )

        # Extract  response
        sentiment = response.choices[0].message.content
        return sentiment
    except Exception as e:
        print(f"Error: {e}")
        return None

# Load human-annotated data
df = pd.read_csv('')

# Apply function and add results to df
df['v0-zeroshot-annotation'] = df['text'].apply(lambda text: zeroshot_sentiment_annotation(text, prompt=sentiment_v0_prompt))
df['v1-zeroshot-annotation'] = df['text'].apply(lambda text: zeroshot_sentiment_annotation(text, prompt=sentiment_v1_prompt))
df['v2-zeroshot-annotation'] = df['text'].apply(lambda text: zeroshot_sentiment_annotation(text, prompt=sentiment_v2_prompt))
df['v3-zeroshot-annotation'] = df['text'].apply(lambda text: zeroshot_sentiment_annotation(text, prompt=sentiment_v3_prompt))

df['v0-fewshot-annotation'] = df['text'].apply(lambda text: fewshot_sentiment_annotation(text, prompt=sentiment_v0_prompt, fewshot_dataset=df))
df['v1-fewshot-annotation'] = df['text'].apply(lambda text: fewshot_sentiment_annotation(text, prompt=sentiment_v1_prompt, fewshot_dataset=df))
df['v2-fewshot-annotation'] = df['text'].apply(lambda text: fewshot_sentiment_annotation(text, prompt=sentiment_v2_prompt, fewshot_dataset=df))
df['v3-fewshot-annotation'] = df['text'].apply(lambda text: fewshot_sentiment_annotation(text, prompt=sentiment_v3_prompt, fewshot_dataset=df))

# Save the results to a new .csv file
df.to_csv('', index=False)

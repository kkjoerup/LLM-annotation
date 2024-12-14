import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

#API key
client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

# load data
data = pd.read_csv('')

# best prompt
topic_v3_prompt = """
Du er en gennemsnitlig dansk nyhedsforbruger. Du får en artikel, og skal tildele den et passende antal kategorier svarende til de emner den omhandler. Artiklen omhandler oftest flere emner, men en kan også kun omhandle ét emne. 
Her er nogle generelle principper, du skal følge: Du skal anvende generel verdensviden og artiklens kontekst, hvor det er nødvendigt. Et emne skal bidrage med at fremføre artiklens væsentligste pointer for at kategorien tildeles. Det er muligt at samtagge flere kategorier, så to kategorier tilsammen skaber det emne, artiklen omhandler.
Kategorier: ”Begivenhed”: En hændelse, der finder sted - særligt for at markere noget vigtigt og evt. tilbagevendende. Herunder mærkedag, personlig begivenhed, sportsbegivenhed og underholdningsbegivenhed. ”Bolig”: Privatpersoners hjem. Herunder køb og salg, renovering/indretning og udlejning. ”Erhverv”: Emner relateret til beskæftigelsesmæssige forhold, inkl. arbejdspladser. Herunder ansættelsesforhold, offentlig instans og privat virksomhed. ”Dyr”: Levende flercellede organismer, ekskl. planter og mennesker. ”Katastrofe”: Begivenheder, som har negative konsekvenser for en større eller mindre gruppe (mennesker). Herunder mindre ulykke og større katastrofe. ”Kendt”: Personer, som ofte er alment kendt blandt den brede befolkning, fx deltagere i tv-programmer, politikere, sportspersonligheder, toperhvervsfolk og kongelige. ”Konflikt og krig”: Sammenstød, uoverensstemmelser eller stridigheder mellem to eller flere parter - evt. med voldelige konsekvenser. Herunder terror og væbnet konflit. ”Kriminalitet”: Ulovligheder, der omhandler fx dramatiske, dødelige eller sindsoprivende begivenheder, hvori politiet evt. har været involveret. Herunder bandekriminalitet, bedrageri og personfarlig kriminalitet. ”Kultur”: Den levevis, som er resultat af menneskelig aktivitet, og forestillingsverden, herunder skikke, holdninger og traditioner, ved en bestemt befolkningsgruppe i en bestemt periode. Herunder byliv, kunst, museum og seværdighed, or rejse. ”Livsstil”: Den bevidst valgte levemåde, som (en) privatperson(er) har - ofte for at signalere en bestemt identitet. Herunder erotik, familieliv, fritid, krop og velvære, mad og drikke og partnerskab. ”Politik”: Samfundsrelevante problemstillinger, værdier o.lign., som udføres på baggrund af en bestemt ideologi. Herunder international politik og national politik. ”Samfund”: Forhold for mennesker, der typisk deler samme geografiske område og/eller er knyttet sammen på anden vis, fx gennem indbyrdes afhængighed og tilknytning til nationalstat. Herunder bæredygtighed og klima, værdier, religion og tendens. ”Sport”: (Fysisk) aktivitet, som udøves individuelt eller på hold i professionelle sammenhænge, og hvor der ofte indgår specifikke regler og evt. udstyr. Herunder cykling, håndbold, fodbold, ketcher- og batsport og motorsport. ”Sundhed”: Helbredsmæssige forhold. Herunder kosmetisk behandling og sygdom og behandling. Teknologi”: Beskrivelser og anvendelser af tekniske hjælpemidler til udførelse af opgaver. Herunder forbrugerelektronik, kunstig intelligens og software. ”Transportmiddel”: Førbare anordninger, som kan fragte levende væsener og/eller artefakter. Herunder bil, mindre transportmiddel, offentlig transport og større transportmiddel. ”Uddannelse”: Strukturerede undervisningsforløb. Herunder grundskole, ungdomsuddannelse og videregående uddannelse. "Underholdning”: Forhold med evne til at more/frembringe glæde hos modtageren. Herunder litteratur, film og tv, musik og lyd og reality. ”Vejr”: Meterologiske forhold, fx forudsigelse eller afrapportering, ofte omhandlende et bestemt geografisk område på et bestemt tidspunkt. "Videnskab”: Metodologisk forskning og undersøgelse af bestemte dele af virkeligheden. Herunder naturvidenskab og samfundsvidenskab og humaniora. ”Økonomi”: Produktion, fordeling og forbrug (hos både privatpersoner og på statsligt niveau) af varer og tjenester. Herunder makroøkonomi og mikroøkonomi. 
Giv et præcist svar i json: {{topics: [”kategori”, ”kategori”, ”kategori”, ... ]}}.
"""

# Function to call GPT-4o with few-shot prompts
def fewshot_topic_annotation(text, prompt, fewshot_dataset):
    # Sample few-shot examples from dataset without current article
    fewshot_dataset = fewshot_dataset[fewshot_dataset['text'] != text]
    fewshot_examples = fewshot_dataset.sample(3).to_dict(orient='records')
    # Create few-shot parrt of prompt
    fewshot_examples = "\n".join([f"Artikel: {example['text']}. Artiklen omhandler dette/disse emne(r): {example['label']}." for example in fewshot_examples])

    try:
        # Make a request to GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {
                    "role": "system", 
                    "content": f"{prompt}. Her er tre eksempler på artikler og deres emner: {fewshot_examples}"
                },
                {
                    "role": "user", 
                    "content": f"Artikel: {text} \nArtiklen omhandler dette/disse emne(r):"
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
                            "topics": {
                                "description": "Topics of the article",
                                "type": "string"
                            },
                            "confidence": {
                                "description": "Confidence score for the topics from 0.00 to 1.00",
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

        # Extract  response
        topics = response.choices[0].message.content
        print('annotated')
        return topics
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# Apply function and add results to df
data['llm_annotation'] = data['text'].apply(lambda text: fewshot_topic_annotation(text, prompt=topic_v3_prompt, fewshot_dataset=data))

# Save the results to a new CSV file
data.to_csv('', index=False)

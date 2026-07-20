# notebooks/test_groq.py
# Groq's API test with Llama3

import os
from dotenv import load_dotenv
from groq import Groq

# Load the key from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("Error : GROQ_API_KEY not found in dotenv")
    exit()

# Create the Groq client
client = Groq(api_key=api_key)

# First call : simple question
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": "Tu es un assistant medical senegalais. Reponds en français simple. Maximum 3 phrases."},
        {"role": "user",
         "content": "Quels sont les symptomes du paludisme ?"}
    ],
    max_tokens=200,
    temperature=0.3
)

# Display the answer
print("=== Reponse de Llama 3 ===")
print(response.choices[0].message.content)
print(f"\nTokens utilises : {response.usage.total_tokens}")



# Test with SenSante format
response2 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": """ Tu es un assistant medical senegalais.
            Tu reçois un diagnostic et des donnees patient.
            Explique le resultat en francais simple ,
            comme un medecin parlerait a son patient .
            Sois rassurant mais recommande une consultation .
            Maximum 3 phrases .
            Ne fais JAMAIS de diagnostic toi - meme ."""},
        {"role": "user",
         "content": """Patient : Femme , 28 ans , region Dakar
            Symptomes : temperature 39.5 , toux , fatigue , maux de tete
            Diagnostic du modele : paludisme ( probabilite 72%)
            Explique ce resultat au patient."""}
    ],
    max_tokens=200,
    temperature=0.3
)

print("=== Explanation SenSante ===")
print(response2.choices[0].message.content)
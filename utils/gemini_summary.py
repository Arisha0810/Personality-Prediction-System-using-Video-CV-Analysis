import google.generativeai as genai
import os

# Configure your Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel('gemini-1.5-pro-latest')

def generate_personality_summary(traits: dict):
    prompt = f"""Generate a personality profile summary based on these Big Five personality traits:
    - Extraversion: {traits['Extraversion']}
    - Neuroticism: {traits['Neuroticism']}
    - Agreeableness: {traits['Agreeableness']}
    - Conscientiousness: {traits['Conscientiousness']}
    - Openness: {traits['Openness']}

    The summary should be concise, human-like, and easy to understand.
    """

    response = model.generate_content(prompt)
    return response.text.strip()

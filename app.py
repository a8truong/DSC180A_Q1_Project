import streamlit as st 
import os
import openai
from dotenv import load_dotenv
from openai import OpenAI

from trulens.core import Feedback, TruSession
from trulens.core.guardrails.base import block_input, block_output
#from trulens.apps.basic import TruBasicApp
from trulens.providers.openai import OpenAI as OAI

#import asyncio
#import nest_asyncio

# Apply nest_asyncio BEFORE importing any async libraries
#nest_asyncio.apply()

# Set up event loop explicitly
#try:
#    loop = asyncio.get_event_loop()
#except RuntimeError:
#    loop = asyncio.new_event_loop()
#asyncio.set_event_loop(loop)


#from nemoguardrails import RailsConfig, LLMRails

#config = RailsConfig.from_path("config/config.yml")
#rails = LLMRails(config)

load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

oai_provider = OAI()

f_hate = Feedback(
    oai_provider.moderation_hate, higher_is_better=False
).on_output()
f_violent = Feedback(
    oai_provider.moderation_violence, higher_is_better=False
).on_output()
f_selfharm = Feedback(
    oai_provider.moderation_selfharm, higher_is_better=False
).on_output()
f_maliciousness = Feedback(
    oai_provider.moderation_harassment_threatening, higher_is_better=False
).on_output()
f_sexual = Feedback(
    oai_provider.moderation_sexual, higher_is_better=False
).on_output()

#feedbacks = [f_hate, f_violent, f_selfharm, f_maliciousness]

def without_guardrails(text):
    response = client.completions.create(
        prompt="Translate the following text to Japanese language:\n"+text,
        model="gpt-3.5-turbo-instruct",
        max_tokens=2048,
        temperature=0)

    result = response.choices[0].text.strip()
    return result

def catch_pii(output: str):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{
            "role": "system",
            "content": """You are a PII detection bot. You will be given text, and you need to find any PII and replace it with <PII_TYPE/>, where the 'PII_TYPE'
              is replaced with the kind of PII being filtered out (for example, PHONE_NUMBER, FULL_NAME, PASSWORD etc...). If no PII is present, you can return it as-is."""
        }, {
            "role": "user",
            "content": "Repeat this text:\n"+output
        }],
        max_tokens=2048,
        temperature=0
    )

    return res.choices[0].message.content

def catch_profanity(output: str):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{
            "role": "system",
            "content": """You are a profanity detection bot. You will be given text to repeat, but you need to first check if
            the text is profanity in the context of the sentence. 
            Example: "I'd kill for a good burger" - not profanity, translate
            
            Example: "Killer dance moves, dude" (slang, 'killer' as adjective) - not profanity, translate

            Example: "You're killing it brother!" (slang) - not profanity, translate

            Example: "I'm going to pet my cat" - not profanity, translate

            Example: "Japanese people" - not profanity, translate

            If there is no profanity, output the text as is.

            If it contains profanity, you should respond with 'Sorry, I can't translate this.'"""
        }, {
            "role": "user",
            "content": "Repeat this text:\n"+output
        }],
        max_tokens=2048,
        temperature=0
    )

    return res.choices[0].message.content

#rails = [catch_pii_llms, catch_profanity]

@block_input(feedback = f_selfharm,
                threshold = 0.5,
                return_value = "Sorry, I can't translate that (Self Harm).")
@block_input(feedback = f_hate,
                threshold = 0.5,
                return_value = "Sorry, I can't translate that (Hate Speech).")
@block_input(feedback = f_violent,
                threshold = 0.1,
                return_value = "Sorry, I can't translate that (Violence).")
@block_input(feedback = f_maliciousness,
                threshold = 0.5,
                return_value = "Sorry, I can't translate that (Malicious).")
@block_input(feedback = f_sexual,
                threshold = 0.5,
                return_value = "Sorry, I can't translate that (Sexual Content).")
def with_guardrails(text):
    g1 = catch_profanity(text)
    g2 = catch_pii(g1)


    #response = rails.generate(messages=[{
    #"role": "user",
    #"content": f"Translate the following text: {text}"
    #    }]).choices[0].message.content
    #guard = Guard().use(
    #CorrectLanguage(expected_language_iso="en", threshold=0.75))

    #guard.validate(response)
    if (g2 == "Sorry, I can't translate this.") and (g2 != text):
        return g2
    response = client.completions.create(
        prompt="Translate the following text to Japanese:\n"+g2,
        model="gpt-3.5-turbo-instruct",
        max_tokens=2048,
        temperature=0)

    result = response.choices[0].text.strip()
    return result

def main():

    st.title("Guardrails Implementation in LLMs")

    text_area = st.text_area("Enter the text to be translated to Japanese")

    if st.button("Translate"):
        if len(text_area)>0:
            st.info(text_area)

            st.warning("Translation Response Without Guardrails")
            without_guardrails_result = without_guardrails(text_area)
            st.success(without_guardrails_result)

            st.warning("Translation With Guardrails")
            with_guardrails_result = with_guardrails(text_area)
            st.success(with_guardrails_result)


if __name__ == "__main__":
    main()


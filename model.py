from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv


def load_model():
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY in .env")
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.4,
        max_output_tokens=512,
        api_key=GEMINI_API_KEY,
    )
    return llm
from dotenv import load_dotenv
import os

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# model = init_chat_model(
#     "gemini-2.5-flash-lite",
#     model_provider="google_genai"
# )

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite"
    
# )

# response = model.invoke("what is cricket?")
# print(response.content)



chat_model = ChatGroq(
    model="llama-3.3-70b-versatile",  
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=20
)

#temperature is a parameter that controls the randomness of the model's output. A higher temperature will result in more random and creative responses, while a lower temperature will produce more deterministic and focused responses.

#max_tokens is a parameter that limits the maximum number of tokens in the generated response. This can help control the length of the output and prevent excessively long responses.

response = chat_model.invoke("give me some info on F1?")
print(response.content)




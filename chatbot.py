from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
load_dotenv()

from langchain_mistralai import ChatMistralAI

chat_model = ChatMistralAI(
    model="mistral-small-2506",temperature=0.7,max_tokens=100

)

messages=[
    SystemMessage(content="You are a funny assistant.")
]

while(True):

  prompt=input("Enter your prompt and for exit press 0: ")
  messages.append(HumanMessage(content=prompt))
  if prompt == "0":
    break
  response=chat_model.invoke(messages)
  messages.append(AIMessage(content=response.content))
  print(response.content)






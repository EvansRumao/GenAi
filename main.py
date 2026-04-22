from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
load_dotenv()


model=ChatMistralAI(
  model="mistral-small-2506"
)
data = PyPDFLoader("document loader/GRU.pdf")

# data = TextLoader("document loader/notes.txt")
docs=data.load()
template=ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant."),
    ("human", "{data}")
  ]
).format_messages(data=docs[0].page_content)



result=model.invoke(template)
print(result.content)


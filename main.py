from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()


model=ChatMistralAI(
  model="mistral-small-2506"
)
# data = PyPDFLoader("document_loader/GRU.pdf")

# data = TextLoader("document_loader/notes.txt")
# docs=data.load()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1000,
#     chunk_overlap = 200
# )

# chunks = splitter.split_documents(docs)


template=ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant."),
    ("human", "{data}")
  ]
)



result=model.invoke(template)
print(result.content)


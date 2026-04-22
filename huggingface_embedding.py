from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
   
)

texts=[
    "This is a test document.",
    "This is another test document.",
    " working on embedding models is fun."
]


response = embedding.embed_documents(texts)
print(response)
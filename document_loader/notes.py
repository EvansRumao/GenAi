from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

# load_dotenv()

data = TextLoader("document_loader/notes.txt")


docs=data.load()

print(docs)
print(docs[0])
print(docs[0].page_content)
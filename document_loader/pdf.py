from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


data = PyPDFLoader("document loader/GRU.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=10
)

chunks = splitter.split_documents(docs)

print(chunks[0].page_content) #content at page 0 is loaded and split into chunks of 1000 characters with an overlap of 10 characters between chunks.
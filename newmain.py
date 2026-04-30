from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader      
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from typer import prompt

load_dotenv()   


model=ChatMistralAI(
  model="mistral-small-2506"
)


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma-db"
)

retriver = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, "fetch_k":10 , "lambda_mult":0.4}
)



template=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

               Use ONLY the provided context to answer the question.

               If the answer is not present in the context,
               say: "I could not find the answer in the document."
          """
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
"""
        )
    ]
)

print("Rag system created ")

print("press 0 to exit ")

while True:
    query = input("You : ")
    if query == "0":
        break 
    
    docs = retriver.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = template.invoke({
        "context" :context,
        "question": query
    })
    
    response = model.invoke(final_prompt)

    print(f"\n AI: {response.content}")




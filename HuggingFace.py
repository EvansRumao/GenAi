from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1"
)

model = ChatHuggingFace(llm=llm_endpoint)

response = model.invoke("give me some info on F1?")
print(response.content)
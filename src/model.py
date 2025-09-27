from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import  HuggingFaceEndpoint,ChatHuggingFace , HuggingFacePipeline
from dotenv import load_dotenv
import os
load_dotenv()

os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={'normalize_embeddings': False}
)



llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    temperature=0.2,
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)
model =ChatHuggingFace(llm=llm)


llm2=HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b',
                         temperature=0.2,
                         huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_API_TOKEN')
)
model2=ChatHuggingFace(llm=llm2)

llm3 = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs={'temperature': 0.2},  
)

model3=ChatHuggingFace(llm=llm3)
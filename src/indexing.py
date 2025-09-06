from langchain.vectorstores import FAISS
from langchain.schema import Document
from chunking import chunking
from model import embedding_model
def indexing():
  c_chunk,r_chunk = chunking()
  constitution_docs = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in c_chunk]
  Right_and_law_docs = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in r_chunk]

  constitution_db = FAISS.from_documents(constitution_docs, embedding_model)
  Right_and_law_db = FAISS.from_documents(Right_and_law_docs, embedding_model)

  constitution_db.save_local("constitution_db")
  Right_and_law_db.save_local("Right_and_law_db")
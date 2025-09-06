from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.vectorstores import FAISS
from indexing import indexing
from model import embedding_model
import os
def Retrieveal(state):
    """
    Combines a multi-query retriever with a cross-encoder reranker.
    If FAISS indices already exist, load them.
    Otherwise, load PDFs, embed, and save the indices.
    """
    query = state['chat_history'][-1].content
    print(query)
    constitution_index_path = "constitution_db"
    rightlaw_index_path = "Right_and_law_db"

    try:
        if not os.path.exists(constitution_index_path) and not os.path.exists(rightlaw_index_path):
            indexing()
    except Exception as e:
        print(f"Error loading or creating FAISS vector stores: {e}")
        return {"docs": []}
    C_vectorstore = FAISS.load_local(
                constitution_index_path, embedding_model, allow_dangerous_deserialization=True
            )
    R_vectorstore = FAISS.load_local(
                rightlaw_index_path, embedding_model, allow_dangerous_deserialization=True
            )
    retrievers = [
        C_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
        R_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
    ]

    multi_query_retriever = EnsembleRetriever(
        retrievers=retrievers,
        weights=[0.5, 0.5],
    )

    rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=rerank_model, top_n=8)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever,
    )

    compressed_docs = compression_retriever.invoke(query)

    return {"docs": compressed_docs}

from langchain.prompts import PromptTemplate
from model import model
def check_relevence_document(state):
    docs = state['docs']
    document_chunk = ''
    for doc in docs:
        document_chunk += doc.page_content

    relevence_document_check = PromptTemplate(
        input_variables=["query", "document_chunk"],
        template='''You are a relevance checker. Your task is to determine if the provided document is relevant to the user's query.

Query: {query}

Document: {document_chunk}

Is the document relevant to the query?
Respond with ONLY one word: "yes" or "no".'''
    )

    chain = relevence_document_check | model
    response = chain.invoke({"query": state['chat_history'][-1].content, "document_chunk": document_chunk})

    return {"is_relevent_document": response.content.strip().lower()}
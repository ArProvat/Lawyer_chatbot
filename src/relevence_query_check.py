from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from model import model
def relevence_query_check(state) -> str:

    class RelevanceCategory(BaseModel):
        category: Literal["relevant", "memory", "irrelevant"] = Field(
            description="Classification of query: relevant, memory, or irrelevant."
        )
    parser = PydanticOutputParser(pydantic_object=RelevanceCategory)

    retrieval_prompt = PromptTemplate(
        input_variables=["query",'memory'],
        template="""
Classify the query into one of the following categories:
1. "relevant" - if it is related to Bangladesh Constitution, human rights, or law

2. "memory" - if it is about the previous conversation: {memory}


3. "irrelevant" - if it is not related to constitution,law or conversation history  

Query: {query}

{format_instructions}
Output should be in JSON format.
""" ,
        partial_variables={"format_instructions": parser.get_format_instructions()},

    )

    chain = retrieval_prompt | model | parser
    print(state['chat_history'][-1])
    response = chain.invoke({"query": state['chat_history'][-1].content,"memory":state['chat_history']})
    print(response)
    return {"is_relevence_query": response.category.strip().lower()}
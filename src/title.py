from model import model3
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import sqlite3
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

def title_generation(context, model3):
    class TitleCategory(BaseModel):
        title: str = Field(
            description="A concise and relevant title for the given context, no more than 3-5 words."
        )
    
    parser = PydanticOutputParser(pydantic_object=TitleCategory)
    
    prompt = PromptTemplate(
        input_variables=["context"],
        template='''You are a title generator. Your task is to generate a concise and relevant title for the given context: {context}
        The title should be no more than 3-5 words and should accurately reflect the content of the context.
        Respond with ONLY the title and no additional text.
        
        {format_instructions}
        ''',
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | model3 | parser
    
    try:
        parsed_response = chain.invoke({"context": context})
        response = parsed_response.title.strip()
        return response
    except Exception as e:
        # It's good practice to handle potential parsing errors
        print(f"An error occurred during title generation: {e}")
        return "Untitled Chat"

def setup_title_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_titles (
            thread_id TEXT PRIMARY KEY,
            title TEXT
        );
    """)
    conn.commit()
    conn.close()

setup_title_db()

def save_chat_title(thread_id, title):
    with sqlite3.connect("chat_history.db") as conn:
        cursor = conn.cursor()
        cursor.execute("REPLACE INTO chat_titles (thread_id, title) VALUES (?, ?)", (thread_id, title))
        conn.commit()

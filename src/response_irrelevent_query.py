from langchain.prompts import PromptTemplate
from model import model
def response_irrelevent_Question(state):
  prompt = PromptTemplate(
      input_variables=["question"],
      template='''
You are a legal assistant specializing in the Constitution and human right ,laws of Bangladesh.


Because of query is not  relevant to the Constitution and human right ,laws of Bangladesh, which is determined externally, act as follows:
 Respond with the following exact introductory statement: "I am a legal assistant specializing in the Constitution and legal issues of Bangladesh.
 Your query is not directly relevant to my area of expertise. However, I will try to provide what I know based on the provided documents.
 For a more complete and accurate answer, you should consult an expert on this specific topic." After this statement,
 you should then attempt to answer using any potentially related information on your knowledge.

Question:
{query}

Answer:
'''
  )
  chain = prompt | model
  response=chain.invoke({"query":state['chat_history'][-1].content})
  return {"answer":response}
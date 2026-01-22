
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, StateGraph
import os
from dotenv import load_dotenv

# Load environment variables (for local development)
load_dotenv()

# --- 1. SETUP KNOWLEDGE BASE (This would be done once on server startup) ---
print("Setting up knowledge base...")
urls = [
    "https://blog.hubspot.com/marketing/what-is-content-marketing",
    "https://www.semrush.com/blog/seo-copywriting/",
    "https://neilpatel.com/blog/how-to-create-a-successful-social-media-marketing-campaign/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="marketing-blogs",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
print("Knowledge base ready.")

# --- 2. DEFINE THE LANGGRAPH AGENT ---
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def retrieve_docs(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def grade_documents(state):
    question = state["question"]
    documents = state["documents"]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {document} \n\n Here is the user question: {question} \n
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant.
        Provide the binary score as a JSON with a single key 'score'.""",
        input_variables=["question", "document"],
    )
    retrieval_grader = prompt | llm | JsonOutputParser()
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        if grade == "yes":
            filtered_docs.append(d)
    return {"documents": filtered_docs, "question": question}

def generate(state):
    question = state["question"]
    documents = state["documents"]
    prompt = PromptTemplate(
        template="""You are a marketing assistant AI. Answer the user's question based on the context below:
        CONTEXT: {context} \n QUESTION: {question} \n ANSWER:""",
        input_variables=["context", "question"],
    )
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def decide_to_generate(state):
    if not state["documents"]:
        return "cannot_answer"
    else:
        return "generate"

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {"generate": "generate", "cannot_answer": END})
workflow.add_edge("generate", END)
app_runnable = workflow.compile()
print("Agent graph compiled and ready.")


# --- 3. SETUP FASTAPI APP ---
app = FastAPI(
  title="Marketing Research Agent API",
  description="An API for running a marketing research agent using Agentic RAG.",
  version="1.0.0",
)

class AgentRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    answer: str

@app.post("/run-agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """
    Runs the marketing research agent on a given query.
    """
    inputs = {"question": request.query}
    result = app_runnable.invoke(inputs)
    return {"answer": result.get("generation", "I could not find relevant information to answer your question.")}

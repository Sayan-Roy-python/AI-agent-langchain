# AI Marketing Research Agent

This project is a lightweight AI agent designed to perform marketing-relevant research tasks. It uses an **Agentic RAG (Retrieval-Augmented Generation)** architecture built with **LangGraph** to answer user queries by searching a curated knowledge base of marketing blogs.

The agent's logic is served via a **FastAPI** backend, fulfilling the core project requirements.

## Core Features

* **Agentic RAG Workflow:** Instead of a simple "retrieve-then-generate" pipeline, this agent uses a LangGraph graph to create a cyclical reasoning loop.
* **Relevance Grading:** The agent retrieves documents and then uses an LLM to "grade" their relevance to the query before deciding whether to use them. This significantly improves response precision and reduces hallucinations.
* **API-Ready:** The entire agent workflow is wrapped in a robust FastAPI server, ready for integration.
* **Interactive Prototype:** A full Google Colab notebook (`run.ipynb`) is included to demonstrate the development, testing, and execution of the agent step-by-step.

## Agent Architecture

The agent operates as a state machine defined in LangGraph:

1.  **Retrieve:** The agent takes the user's query and retrieves relevant document chunks from the ChromaDB vector store.
2.  **Grade Documents:** The agent iterates through each retrieved document and uses an LLM call to decide if it is ("yes" or "no") relevant to the original query.
3.  **Conditional Edge (Decide):** Based on the grades, the agent decides:
    * If relevant documents are found, it proceeds to the `generate` node.
    * If no relevant documents are found, it ends the process (preventing hallucinated answers).
4.  **Generate:** The agent synthesizes a final, coherent answer using only the filtered, relevant documents as context.

## Technology Stack

* **Backend:** FastAPI
* **Orchestration:** LangGraph & LangChain
* **LLM:** OpenAI (GPT-4o)
* **Vector Database:** ChromaDB
* **Data Handling:** Pydantic

## Project Files

This repository contains two primary components:

1.  **`run.ipynb`**: A Google Colab notebook that serves as the interactive "workbench." Use this to understand how the agent is built, test individual components, and experiment with the logic.
2.  **`main.py`**: The final, production-ready Python application. This single file contains all the code needed to initialize the knowledge base, define the agent, and run the FastAPI server. This file is generated and exported from the Colab notebook.

---

## Setup and Installation

### Prerequisites

* Python 3.10+
* An OpenAI API Key

### Local Setup

1.  **Clone the repository (or download the files):**


2.  **Install dependencies:**
    (can install manually)
    ```bash
    pip install "langgraph" "langchain" "langchain_openai" "chromadb" "fastapi" "uvicorn[standard]" "python-dotenv" "beautifulsoup4" "lxml"
    ```

3.  **Create an environment file:**
    Create a file named `.env` in the root of the project and add your API key:
    ```ini
    OPENAI_API_KEY="sk-..."
    ```

---

## How to Run

You can run this project in two ways:

### Option 1: Run the Interactive Prototype (Recommended for Testing)

1.  Upload `run.ipynb` to Google Colab.
2.  Add your OpenAI API key to the Colab Secrets Manager (under the key icon) with the name `OPENAI_API_KEY`.
3.  Run the cells from top to bottom to see the agent being built and tested in real-time.

### Option 2: Run the FastAPI Server (Production)

1.  Make sure you have completed the **Local Setup** steps above.
2.  From your terminal, run the Uvicorn server:

    ```bash
    uvicorn main:app --reload
    ```

3.  The API will be live and accessible at `http://127.0.0.1:8000`.

## API Usage

Once the FastAPI server is running, you can interact with the agent via the `/run-agent` endpoint.

### API Docs

Interactive documentation (via Swagger UI) is automatically available at:
**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### cURL Example

and its expected response:
 {
  "answer": "SEO copywriting plays a crucial role in content marketing by creating content that is optimized for search engines while also being engaging and valuable to the human reader. Its goal is to increase organic traffic by ranking higher in search results for specific keywords, ultimately driving that traffic to convert."
}

Here is a sample request using cURL:

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/run-agent](http://127.0.0.1:8000/run-agent)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What is the role of SEO copywriting in content marketing?"
}'

import os
import re
import json
from typing import List, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import PGVector
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain_core.documents import Document
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from graph_state import GraphState

load_dotenv()

# --- Initialize Core Components ---
CONNECTION_STRING = os.getenv('DATABASE_URL')
LLM_MODEL_PATH = "./models/mistral-7b-v0.1.Q4_K_M.gguf"

# Define PGVector collection names
SCHEMA_COLLECTION_NAME = "my_database_schema"
EXAMPLES_COLLECTION_NAME = "my_sql_examples"

# Instantiate embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Instantiate retrievers for both schema and examples from PGVector
schema_vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=SCHEMA_COLLECTION_NAME,
    embedding_function=embeddings,
)
schema_retriever = schema_vectorstore.as_retriever(search_kwargs={"k": 5})

examples_vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=EXAMPLES_COLLECTION_NAME,
    embedding_function=embeddings,
)
examples_retriever = examples_vectorstore.as_retriever(search_kwargs={"k": 3})


# Instantiate local LLM
llm = LlamaCpp(
    model_path=LLM_MODEL_PATH,
    n_gpu_layers=1,
    n_ctx=4096,
    n_batch=512,
    verbose=True,
)

# --- Node 1: Schema and Example Retrieval ---
def retrieve_schema(state: GraphState):
    """Retrieves relevant schema and examples from the vector store."""
    print("---RETRIEVING CONTEXT (SCHEMA & EXAMPLES)---")
    user_question = state["user_question"]
    
    # Retrieve relevant schema
    schema_docs = schema_retriever.invoke(user_question)
    schema = "\n".join([doc.page_content for doc in schema_docs])

    # Retrieve relevant examples
    example_docs = examples_retriever.invoke(user_question)
    examples = "\n\n".join([doc.page_content for doc in example_docs])
    
    updated_state = state.copy()
    updated_state["retrieved_schema"] = schema
    updated_state["retrieved_examples"] = examples
    
    return updated_state

# --- Node 2: SQL Generation ---
def generate_sql(state: GraphState):
    """Generates an SQL query using the LLM based on the question, schema, and examples."""
    print("---GENERATING SQL QUERY---")
    
    user_question = state.get("user_question", "")
    schema = state.get("retrieved_schema", "")
    examples = state.get("retrieved_examples", "")

    # --- SIMPLIFIED PROMPT TEMPLATE ---
    prompt_template = PromptTemplate.from_template(
        f"""You are a PostgreSQL expert. Your sole purpose is to generate a single, valid, and correct PostgreSQL query based on the user's question, the provided database schema, and the examples.
        
        You MUST ONLY output the SQL query and nothing else. Do not add any conversational text, explanations, or code delimiters.
        
        ### Database Schema
        {schema}
        
        ### Examples
        {examples}

        ### User Question
        {user_question}

        ### SQL Query:
        """
    )
    
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    response = llm_chain.invoke({"schema": schema, "user_question": user_question, "examples": examples})

    generated_text = response['text']
    
    # --- SIMPLIFIED EXTRACTION LOGIC ---
    # The LLM's response starts after "### SQL Query:", so we can split there.
    # We no longer need the regex since we've removed the delimiters.
    sql_query = generated_text.split("### SQL Query:")[-1].strip()
    
    # A common issue is the LLM adding extra newline characters or spaces.
    # Ensure a single line for the query.
    sql_query = sql_query.split('\n')[0].strip()

    updated_state = state.copy()
    updated_state["sql_query"] = sql_query
    
    return updated_state

# --- Node 3: Validation and Execution ---
def validate_and_execute(state: GraphState):
    """Validates the generated query and executes it against the database."""
    print("---VALIDATING AND EXECUTING SQL---")
    sql_query = state.get("sql_query", "")
    
    if not sql_query.lower().strip().startswith("select"):
        updated_state = state.copy()
        updated_state["error_message"] = "Only SELECT queries are allowed. Query rejected."
        return updated_state
        
    try:
        engine = create_engine(CONNECTION_STRING)
        with engine.connect() as connection:
            result = connection.execute(text(sql_query)).fetchall()
            updated_state = state.copy()
            updated_state["query_result"] = [str(row) for row in result]
            updated_state["error_message"] = None
    except Exception as e:
        updated_state = state.copy()
        updated_state["error_message"] = str(e)
        updated_state["query_result"] = []
    
    return updated_state
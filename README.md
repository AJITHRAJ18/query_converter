# SQL Generation with Retrieval-Augmented Generation (RAG)

This project demonstrates an intelligent system that translates natural language questions into SQL queries using a Retrieval-Augmented Generation (RAG) framework. By leveraging a vector store and a Foundation Model, this workflow can accurately generate SQL queries based on a given database schema and relevant examples, even for complex or specific requests.

## Workflow

The system operates through a multi-step pipeline to ensure a high degree of accuracy and reliability. The process begins with a user's natural language input and concludes with an executable SQL query or a result.

1) User Input
- A user provides a natural language question (e.g., "how many products are there?").

2) Prompt Templating
- The user input is combined with a predefined prompt template, which provides context and instructions for the Foundation Model.

3) Context Retrieval (RAG)
- Schema Embedding: Database table and schema definitions are embedded and stored in a vector store.
- Example Embedding: Sample questions with their corresponding, correct SQL queries are also embedded and stored.
- Final Prompt Generation: Based on the user's question, the most relevant schema definitions and examples are retrieved from the vector store. This retrieved context, along with the user's question, forms the final prompt.

4) SQL Generation
- The final prompt is fed into a Foundation Model (e.g., Mistral-7B). The model acts as a "PostgreSQL expert" and generates an SQL statement.

5) Validation & Error Handling
- The generated SQL statement is run through a validation step to ensure it is a valid SELECT query. Exceptions (like UndefinedTable or UndefinedColumn) are caught here, preventing incorrect queries from being executed.

6) Database Execution
- The validated SQL statement is executed against the target database.

7) Result
- The final result of the query is returned to the user.

## Workflow Diagram

<img width="4584" height="1644" alt="image" src="https://github.com/user-attachments/assets/df383435-c670-4ba6-aa23-6d38eab67454" />


## Core Components

- Foundation Model
  - A large language model (e.g., Mistral-7B) responsible for translating the final prompt into a coherent SQL query.
- Vector Store
  - A database (e.g., PGVector) used to store embedded representations of the database schema and example queries.
- Prompt Template
  - A structured text format that guides the LLM, providing it with identity, rules, and context.
- Retrieval System
  - The mechanism (e.g., LangGraph) that fetches the most relevant schema and examples from the vector store to create the final prompt.

## Features

- Natural language to SQL generation with schema-aware retrieval
- Example-augmented prompting (few-shot) for higher accuracy
- Validation layer to block unsafe/invalid SQL
- Error-aware regeneration loop leveraging exception feedback
- Pluggable components for model, retriever, and vector store

## Requirements

- Python 3.10+
- PostgreSQL (for execution and PGVector)
- Access to an LLM provider (e.g., Mistral API) and API key
- Optional: PGVector extension or an alternative vector database


## Typical Workflow

1. Export schema metadata to JSON.
2. Ingest schema + example Q&A into the vector store.
3. Convert a question to SQL.
4. Validate and execute.



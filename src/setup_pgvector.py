import os
from sqlalchemy import create_engine, text
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from schema_extractor import get_db_schema_string
from dotenv import load_dotenv
import json

load_dotenv()

# Define your database connection details
CONNECTION_STRING = os.getenv('DATABASE_URL')
COLLECTION_NAME = "my_database_schema"
EXAMPLES_COLLECTION_NAME = "my_sql_examples"

def setup_pgvector_store():
    """
    Sets up the PGVector store with your database schema embeddings.
    """
    print("✨ Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("✅ Embeddings model loaded.")

    # 1. Get the database schema string
    print("🔍 Extracting schema from the database...")
    try:
        schema_string = get_db_schema_string()
    except Exception as e:
        print(f"❌ Failed to get schema: {e}")
        return

    # 2. Split the schema into individual table definitions and create documents
    table_definitions = [doc for doc in schema_string.split(';') if doc.strip()]
    docs = [Document(page_content=table_def.strip(), metadata={}) for table_def in table_definitions]

    # 3. Connect and populate the PGVector store
    print("📦 Initializing and populating the PGVector store...")
    
    # This will create a table named 'langchain_pg_embedding' if it doesn't exist
    PGVector.from_documents(
        embedding=embeddings,
        documents=docs,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True # Deletes existing collection to avoid duplicates
    )
    
    print(f"🎉 PGVector store for collection '{COLLECTION_NAME}' populated successfully!")

    print(f"📦 Initializing and populating the PGVector store for '{EXAMPLES_COLLECTION_NAME}'...")
    
    # 1. Load the examples from the JSON file
    with open('examples.json', 'r') as f:
        examples = json.load(f)
    
    # 2. Create documents from the examples
    example_docs = [
        Document(
            page_content=f"Question: {ex['question']}\nSQL: {ex['sql']}",
            metadata={"source": "examples"}
        ) for ex in examples
    ]
    
    # 3. Populate the new collection
    PGVector.from_documents(
        embedding=embeddings,
        documents=example_docs,
        collection_name=EXAMPLES_COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True
    )
    print(f"🎉 PGVector store for examples populated successfully!")


if __name__ == "__main__":
    setup_pgvector_store()
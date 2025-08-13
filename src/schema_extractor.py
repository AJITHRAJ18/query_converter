import os
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_db_schema_string() -> str:
    """Connects to the database and returns a string of its schema."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set.")

    try:
        engine = create_engine(db_url)
        inspector = inspect(engine)
        schema_string = "### Database Schema\n\n"
        
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            column_defs = [f"{col['name']} {col['type']}" for col in columns]
            schema_string += f"CREATE TABLE {table_name} ({', '.join(column_defs)});\n"
            
        print("✅ Successfully extracted database schema.")
        return schema_string

    except Exception as e:
        raise ConnectionError(f"❌ Failed to connect to the database or extract schema: {e}")

if __name__ == "__main__":
    try:
        # This block demonstrates how the function would be used
        schema = get_db_schema_string()
        print("\n--- Extracted Schema ---\n")
        print(schema)
    except (ValueError, ConnectionError) as e:
        print(e)
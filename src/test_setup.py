import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


def test_database_connection():
    try:
        engine = create_engine(os.getenv('DATABASE_URL'))
        with engine.connect() as connection:
            result = connection.execute(text('SELECT 1'))
            print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False


    
def test_llm_loading():
    try:
        # Define the path to your local GGUF model
        model_path = "models/mistral-7b-v0.1.Q4_K_M.gguf"

        # Check if the model file exists
        if not os.path.exists(model_path):
            print(f"❌ Local model file not found: {model_path}")
            return False

        # Set up a callback manager for streaming output (optional but useful)
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.1,
            max_new_tokens=2000,
            n_ctx=4096,  # Context window size
            n_gpu_layers=-1, # Set this to -1 to use all GPU layers if available
            callback_manager=callback_manager,
            verbose=True,  # Enable verbose output for debugging
        )
        
        # Invoke the model to test it
        result = llm.invoke("Test")
        print("✅ LLM loaded successfully from local GGUF model")
        return True
    except Exception as e:
        print(f"❌ LLM loading failed: {str(e)}")
        return False

def test_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        test_embedding = embeddings.embed_query("Test query")
        print("✅ Embeddings model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Embeddings loading failed: {str(e)}")
        return False

def main():
    print("Running setup tests...")
    print("-" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Run tests
    db_status = test_database_connection()
    llm_status = test_llm_loading()
    embeddings_status = test_embeddings()
    
    print("-" * 50)
    if all([db_status, llm_status, embeddings_status]):
        print("🎉 All components are working correctly!")
    else:
        print("⚠️ Some components need attention!")

if __name__ == "__main__":
    main()

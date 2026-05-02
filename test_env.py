from dotenv import load_dotenv
import os

load_dotenv()

# Test existing variables
print("OPENAI_API_KEY loaded:", os.getenv("OPENAI_API_KEY") is not None)

# Test non-existent variable (will show None)
print("YOUR_ENV_VAR:", os.getenv("YOUR_ENV_VAR"))
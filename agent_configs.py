# This file contains important stable variables for the agent
import os
from dotenv import load_dotenv

load_dotenv()


# Cohere
COHERE_MODEL = os.getenv("COHERE_MODEL")
COHERE_SYSTEM_PROMPT = """
You are a helpful assistant for answering questions about statistics.
"""


# Baseten
BASETEN_MODEL_ID = os.getenv("BASETEN_MODEL_ID")
BASETEN_SYSTEM_PROMPT = """

You are a helpful assistant for answering questions about statistics.
You answer by transforming input text into specific schemas.
Your job is to take input from the user along with a schema, \
 and transform the input into the schema.

"""

EXAMPLE_TEST_PROMPT = """
Create an example applying the {test_name} to the following situation:
Situation: {situation}
Return the example in JSON format, respecting the TestExample structure.
"""

FIND_TEST_PROMPT = """
Given the following situation, find a set of appropriate statistical tests to apply:
Situation: {situation}
Return the assumptions for each test, whether they pass or not, \
and recommend a test/procedure in JSON format.
"""


# Pinecone
STATWIKI_INDEX = os.getenv("STATWIKI_INDEX")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
TOP_K = int(os.getenv("TOP_K"))

# llamaindex specifics

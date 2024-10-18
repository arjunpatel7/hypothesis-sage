from pinecone import Pinecone
from typing import Optional, Type, Dict, Any
import cohere
from llama_index.llms.cohere import Cohere
import requests
import json
from pydantic import ValidationError, BaseModel

# our own stuff
from stat_structures import TestExample, FindTestResponse

# agent configs
from agent_configs import (
    STATWIKI_INDEX,
    EMBEDDING_MODEL,
    TOP_K,
    COHERE_MODEL,
    COHERE_SYSTEM_PROMPT,
    BASETEN_MODEL_ID,
    BASETEN_SYSTEM_PROMPT,
    EXAMPLE_TEST_PROMPT,
    FIND_TEST_PROMPT,
    PINECONE_API_KEY,
    COHERE_API_KEY,
    BASETEN_API_KEY,
)

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)


# Cohere setup
# modify tool choice if it doesn't work...
llm_cohere = Cohere(model="command-r-plus", api_key=COHERE_API_KEY)
co = cohere.Client(api_key=COHERE_API_KEY)


# Baseten Setup

# import basemodel


def get_baseten_response(
    query: str, context: str, json_structure: Type[BaseModel]
) -> Dict[str, Any]:
    """
    Get a response from the Baseten API, using a specified json schema.
    This is useful for structured generation of text into our assumptions
    This is specifically for only getting structured responses


    """

    payload = {
        "messages": [
            {"role": "system", "content": BASETEN_SYSTEM_PROMPT},
            (
                {"role": "user", "content": "The context for the query is: " + context}
                if context
                else None
            ),
            {"role": "user", "content": query},
        ],
        "max_tokens": 8192,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": json_structure.model_json_schema()},
        },
    }
    url = f"https://model-{BASETEN_MODEL_ID}.api.baseten.co/production/predict"
    headers = {"Authorization": f"Api-Key {BASETEN_API_KEY}"}
    resp = requests.post(url, json=payload, headers=headers)

    # Add error handling and logging
    try:
        resp.raise_for_status()  # This will raise an exception for HTTP errors

        # Print the raw response for debugging
        # print("Raw API response:", resp.text)

        # Try to parse the JSON response
        json_response = resp.json()

        # Validate the response against the expected structure
        return json_structure(**json_response)
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print("Raw response content:", resp.text)
        raise
    except ValidationError as e:
        print(f"Response doesn't match expected structure: {e}")
        raise


def get_cohere_response(
    query, json_structure=None, context=None, enforce_json_structure=False
):
    """
    Get a response from the Cohere API, using a specified json schema.
    This is useful for reformatting responses into specific structures.
    """
    system_prompt = COHERE_SYSTEM_PROMPT
    response_format = {"type": "text"}

    if context:
        system_prompt += f"The context for the query is: {context}"

    if enforce_json_structure:
        if json_structure:
            system_prompt += " You must generate syntactically correct JSON."
            schema = json_structure.model_json_schema(mode="serialization")
            # Check the depth of the schema
            depth = get_json_depth(schema)
            # console.print(f"[bold blue]Schema Depth:[/bold blue] {depth}")
            if depth > 5:
                raise ValueError("JSON schema exceeds the maximum depth of 5 levels")
            response_format = {"type": "json_object", "schema": schema}
        else:
            raise ValueError(
                "json_structure must be provided when enforce_json_structure is True"
            )

    # Debug print to verify response_format
    # console.print(f"[bold blue]ResponseFormat:[/bold blue] {response_format}")

    response = co.chat(
        model=COHERE_MODEL,
        # replaces Cohere system prompt
        preamble=system_prompt,
        message=query,
        response_format=response_format,
        temperature=0.1,
    )

    response_content = response.text
    # this allows for rewriting if json structure is invalid.
    if response_format["type"] == "json_object":
        if validate_json_structure(response_content, json_structure):
            return response_content
        else:
            raise ValueError("Invalid JSON structure")
    else:
        return response_content


def get_json_depth(d, level=1):
    """
    Helper function to determine the depth of a JSON-like dictionary.
    """
    if not isinstance(d, dict) or not d:
        return level
    return max(get_json_depth(v, level + 1) for v in d.values())


def validate_json_structure(response, json_structure):
    """
    Validate a response against a specified json structure.
    """
    try:
        json_structure.model_validate_json(response)
        return True
    except Exception as e:
        print(e)
        return False


def explain(query):
    """
    Query the Pinecone DB for information.
    If the agent can't answer based on query results, return 'I don't know.'
    Useful for explaining concepts, or providing context for other tools.
    """
    # from response, collect the context
    response = query_db(query)
    context = " ".join(
        [item["metadata"]["Chunk Content"] for item in response["matches"]]
    )

    response = get_cohere_response(query=query, context=context)

    return response


def query_db(query):
    query_embedding = pc.inference.embed(
        EMBEDDING_MODEL,
        inputs=[query],  # Use the query for embedding
        parameters={"input_type": "query", "truncate": "END"},
    )
    query_vector = query_embedding[0]["values"]
    index = pc.Index(STATWIKI_INDEX)

    # Add metadata filter to exclude Article Titles containing "Template Talk"
    response = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
    )
    # filter the response to exclude articles with "Template talk" in the title

    return response


def find_test(situation):
    """
    Given a situation, find a set of appropriate statistical tests to apply.
    Useful for finding tests for a given situation.
    Queries the database for relevant tests,
    and attempts to answer with the tests found.
    """

    prompt = FIND_TEST_PROMPT.format(situation=situation)

    # query the db for tests, and add them to the response
    tests = query_db(
        f"Find statistical tests related to {situation}. \
        They should be able to answer the question."
    )
    context = " ".join([item["metadata"]["Chunk Content"] for item in tests["matches"]])

    prompt += f"Here are the tests we found: {context}"

    response = get_baseten_response(
        prompt, context=context, json_structure=FindTestResponse
    )

    try:
        # Validate the response
        validated_response = FindTestResponse.model_validate(response)
        return validated_response
    except ValidationError as e:
        print(f"Validation error: {e}")
        raise


def test_example(test_name: str, situation: Optional[str] = None):
    """
    Create an example applying the given test or procedure.
    Also allows for a situation to be provided.
    Useful for generating test examples for the LLM to use.
    Queries the database for additional context if needed.

    Returns a TestExample instance.
    """

    additional_info = explain(
        f"Provide context for {test_name}"
    )  # Query for additional context

    prompt = EXAMPLE_TEST_PROMPT.format(
        test_name=test_name,
        situation=situation
        or "A hypothetical scenario where we need to apply the test.",
    )

    response = get_baseten_response(
        query=prompt, context=additional_info, json_structure=TestExample
    )

    try:
        # Validate the response
        validated_response = TestExample.model_validate(response)
        return validated_response
    except ValidationError as e:
        print(f"Validation error: {e}")
        raise


## Workflow Code

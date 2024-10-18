from agent_configs import (
    COHERE_MODEL,
    COHERE_API_KEY,
    PHOENIX_API_KEY,
)
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
)
from typing import Optional, Any
from hsage import (
    query_db,
    find_test,
    test_example,
    explain,
)
import cohere
import asyncio
import llama_index.core
import os
from hsage_cli import pretty_print_example, pretty_print_tests


os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
llama_index.core.set_global_handler(
    "arize_phoenix", endpoint="https://llamatrace.com/v1/traces"
)

# Initialize Cohere client
co = cohere.Client(api_key=COHERE_API_KEY)


# Define custom events
class ToolCallEvent(Event):
    """
    Event for tool calls.
    """

    tool_name: str
    arguments: dict
    from_longer_workflow: Optional[bool] = False


class TestRecommendationEvent(Event):
    """
    Event for test recommendations.
    """

    situation: str


class ExampleCreationEvent(Event):
    """
    Event for example creation.
    """

    test_name: str
    situation: Optional[str] = None


class FinalResponseEvent(Event):
    """
    Event for final responses.
    """

    response: str


class HelperEvent(Event):
    """
    Helps pass around tool call results that are not StopEvents
    We use this to collect results from async tool calls for
    Test Recommendation and Example Creation

    TODO: figure out the best way to handle this... maybe via context?

    """

    result: Any
    # additional_args: Optional[dict] = None


# Define available tools
TOOLS = {
    "query_db": query_db,
    "find_test": find_test,
    "test_example": test_example,
    "explain": explain,
}

# import function tool

ASYNC_TOOLS = {
    "async_test_example": FunctionTool.from_defaults(
        fn=test_example,
        description="Creates a single example test \
            for a given test name and situation.",
    )
}

cohere_tools = [
    {
        "name": "query_db",
        "description": "Queries a database using an embedding model and \
        returns relevant results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search \
                        for in the database.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_test",
        "description": "Finds a test for a given situation.",
        "parameters": {
            "type": "object",
            "properties": {
                "situation": {
                    "type": "string",
                    "description": "The situation to find a test for.",
                }
            },
            "required": ["situation"],
        },
    },
    # test example tool
    {
        "name": "test_example",
        "description": "Creates a single example test for a \
            given test name and situation.",
        "parameters": {
            "type": "object",
            "properties": {
                "test_name": {
                    "type": "string",
                    "description": "The name of the test to create an example for.",
                },
                "situation": {
                    "type": "string",
                    "description": "The situation to create an example for.",
                },
            },
            "required": ["test_name", "situation"],
        },
    },
    # explain tool
    {
        "name": "explain",
        "description": "Explains a statistical concept described by the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to explain.",
                }
            },
            "required": ["query"],
        },
    },
    # mock Example Creation Event tool
    # if this is called, we return the ExampleCreationEvent, which will do a workflow
    # which creates several auto-checked examples for a given test name and situation.
    {
        "name": "make_lots_of_examples",
        "description": "Creates several examples for a given test name and situation.",
        "parameters": {
            "type": "object",
            "properties": {
                "test_name": {
                    "type": "string",
                    "description": "The name of the test to create examples for.",
                },
                "situation": {
                    "type": "string",
                    "description": "The situation to create examples for.",
                },
            },
            "required": ["test_name"],
        },
    },
]


def ask_cohere_for_tool_call(query: str, cohere_tools) -> str:
    # only helper for inteacting with Cohere

    response = co.chat(message=query, model=COHERE_MODEL, tools=cohere_tools)

    # parse the response, and return the name.

    return response.tool_calls


def decide_initial_workflow_tool(query: str) -> ToolCallEvent | ExampleCreationEvent:
    """
    Hack for creating a router for the StatisticsWorkflow

    Idea here is that we can pass a set of tools to a LLM
    and ask it to decide which one to use
    But, that can be tricky without a reliable router,
    so we're gonna pretend that the LLM
    is getting a list of tools + workflow "tools", and just using the output
    as a classifier.

    The input will be a query, and the return will be the
    string mapping to the event we need.
    """

    tool_result = ask_cohere_for_tool_call(query, cohere_tools)

    # TODO: handle multiple tool calls and global message context

    tool_name = tool_result[0].name

    if tool_name == "make_lots_of_examples":
        # sometimes situation is not in the arguments, and sometimes it is.
        situation = tool_result[0].parameters.get("situation", "")
        test_name = tool_result[0].parameters.get("test_name", "")
        return ExampleCreationEvent(test_name=test_name, situation=situation)
    else:
        tool_args = tool_result[0].parameters
        return ToolCallEvent(tool_name=tool_name, arguments=tool_args)


# import the pretty printing functions from hsage_cli

# create dictionary to map tool names to pretty printing functions
tool_name_to_pretty_printer = {
    "test_example": pretty_print_example,
    "find_test": pretty_print_tests,
}


# Define the workflow
class StatisticsWorkflow(Workflow):
    @step
    async def router(
        self, ctx: Context, ev: StartEvent
    ) -> ToolCallEvent | ExampleCreationEvent:
        # how do I get the LLM to trigger a tool call OR exa
        query = ev.query
        event = decide_initial_workflow_tool(query)
        return event

    @step(num_workers=10)
    async def tool_calling_step(
        self, ctx: Context, ev: ToolCallEvent
    ) -> StopEvent | HelperEvent:
        # TODO: handle multiple tool calls and global message context,
        # so we won't stop immediately after the first tool call.
        """
        This is sloppy, but is useful because it just allows users
        to access a tool quickly and go.
        """
        tool_name = ev.tool_name
        tool_arguments = ev.arguments
        tool_result = await asyncio.to_thread(TOOLS[tool_name], **tool_arguments)

        if ev.from_longer_workflow:
            return HelperEvent(result=tool_result)
        else:
            if tool_name == "test_example":
                console, table = pretty_print_example(tool_result)
                console.print(table)
            if tool_name == "find_test":
                console = pretty_print_tests(tool_result)
                console.print()
            return StopEvent(result=tool_result)

    @step
    async def example_generation_step(
        self, ctx: Context, ev: ExampleCreationEvent
    ) -> ToolCallEvent:
        """
        This helps users get multiple generated examples for
        how to use a specific test and function.

        We want some more complicated analysis logic
        here when creating the examples, so we create
        a separate event type and step for it.

        Ideally, we could create a TON of examples,
        and throw away the bad or malformed ones
        We can use an LLM that leverages a checklist
        of sorts to ensure examples are useful.

        We're borrowing ideas from Modal's whitepaper on scaling inference
        for accurate results, and
        also a checklist paper released by Cohere recently.

        checklist paper: https://arxiv.org/pdf/2410.03608
        modal whitepaper: https://modal.com/blog/llama-human-eval
        """
        for _ in range(5):
            ctx.send_event(
                ToolCallEvent(
                    tool_name="test_example",
                    arguments={"test_name": ev.test_name, "situation": ev.situation},
                    from_longer_workflow=True,
                )
            )
        return None

    @step
    async def collect_examples(self, ctx: Context, ev: HelperEvent) -> StopEvent:
        # we'll collect the results in a list
        collected_events = ctx.collect_events(ev, [HelperEvent] * 5)
        # TODO: FILTERING STEP
        # we'll use an LLM to filter out the bad examples and keep the good ones
        # for now, we'll just return the first three
        good_examples = []
        if collected_events is None:
            return None
        good_examples = collected_events[:3]

        for example in good_examples:
            console, table = pretty_print_example(example.result)
            console.print(table)

        return StopEvent(result=good_examples)


async def main(query: str):
    w = StatisticsWorkflow(timeout=240, verbose=False)
    result = await w.run(query=query)
    print(result)


if __name__ == "__main__":

    # run the workflow
    query = "make me a bunch of examples for \
    applying statistical tests to Youtube Videos on a given channel"
    # query = "how do I test if two categorical variables are independent?"
    asyncio.run(main(query=query))

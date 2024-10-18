# This file contains the command line interface for Hypothesis Sage
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from hsage import find_test, test_example, explain, query_db
from stat_structures import FindTestResponse, TestExample
import typer
import os
import llama_index.core
from agent_configs import PHOENIX_API_KEY, COHERE_MODEL
from typing import Optional


# Rich Setup
# rich setup
console = Console()

# Phoenix setup
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
llama_index.core.set_global_handler(
    "arize_phoenix", endpoint="https://llamatrace.com/v1/traces"
)


def create_console_from_response(response):
    """
    Create a Rich console from a Pinecone Query response. It looks niiiiiice
    """
    console = Console()

    # Extract relevant information from the response
    titles = [item["metadata"]["Article Title"] for item in response["matches"]]
    chunk_contents = [item["metadata"]["Chunk Content"] for item in response["matches"]]
    scores = [item["score"] for item in response["matches"]]

    # Create a table to display the results
    table = Table(title="Pinecone Query Results")
    table.add_column("Article Title", style="green")
    table.add_column("Chunk Content", style="cyan")
    table.add_column("Score", style="magenta")

    for title, content, score in zip(titles, chunk_contents, scores):
        table.add_row(title, content, str(score))

    return console, table


def pretty_print_example(example: TestExample):
    """Pretty print the TestExample using Rich with markdown support."""
    console = Console()

    # Create a table to display the example details
    table = Table(
        title=f"Test Example: {example.test_name}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    # Function to render markdown content
    def render_markdown(content: str) -> str:
        md = Markdown(content)
        with console.capture() as capture:
            console.print(md)
        return capture.get()

    # Add rows for each field in the TestExample
    table.add_row("Situation", render_markdown(example.situation))
    table.add_row("Test Name", render_markdown(example.test_name))
    table.add_row("Description", render_markdown(example.description))
    table.add_row("Check Assumptions", render_markdown(example.check_assumptions))
    table.add_row("Apply Test", render_markdown(example.apply_test))
    table.add_row("Notes", render_markdown(example.notes))

    # Add assumptions to the table with emojis
    if example.assumption_descriptions and example.assumption_pass_statuses:
        assumptions_str = "\n".join(
            [
                f"{'✅' if pass_status else '❌'} {render_markdown(description)}"
                for description, pass_status in zip(
                    example.assumption_descriptions, example.assumption_pass_statuses
                )
            ]
        )
        table.add_row("Assumptions", assumptions_str)
    else:
        table.add_row("Assumptions", "None")

    return console, table


def pretty_print_tests(find_test_response: FindTestResponse):
    """Pretty print the FindTestResponse using Rich."""
    console = Console()

    # Create a table to display the situation
    situation_table = Table(title="Situation")
    situation_table.add_column("Situation", style="cyan")
    situation_table.add_row(find_test_response.situation)
    console.print(situation_table)

    # Create a table for each recommended test
    for test_recommendation in find_test_response.recommended_tests:
        table = Table(title=f"Test Recommendation: {test_recommendation.test_name}")
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        # Add rows for each field in the TestRecommendation
        table.add_row("Test Name", test_recommendation.test_name)

        # Add assumptions to the table with emojis
        if (
            test_recommendation.assumptions_descriptions
            and test_recommendation.assumptions_pass_statuses
        ):
            assumptions_str = "\n".join(
                [
                    f"{'✅' if pass_status else '❌'} {description}"
                    for description, pass_status in zip(
                        test_recommendation.assumptions_descriptions,
                        test_recommendation.assumptions_pass_statuses,
                    )
                ]
            )
            table.add_row("Assumptions", assumptions_str)
        else:
            table.add_row("Assumptions", "None")

        console.print(table)

    return console


app = typer.Typer()


@app.command()
def query(q: str):
    """
    Query a database of statistical information.
    Useful for providing context for other tools.
    Returns a set of possible chunks that could be important for the LLM.

    """
    response = query_db(q)
    cs, table = create_console_from_response(response)
    cs.print(table)
    return response


@app.command()
def make_example(test_name: str, situation: Optional[str] = None):
    """
    This tool is great for creating examples of tests or procedures a user asks for.
    Optionally, provide a situation to create a TestExample for a specific scenario.
    Also pretty prints this information directly to the console.

    """
    example = test_example(test_name, situation)

    console, table = pretty_print_example(example)
    console.print(table)
    return example


@app.command()
def find_best_test(prompt: str):
    """
    Find the best statistical test for the given prompt.
    Useful for finding the best test for a given situation.
    """
    test_response = find_test(prompt)
    console = pretty_print_tests(test_response)
    console.print()
    return test_response
    # typer.echo(test_response)


@app.command()
def explain_this(query: str):
    """
    Explain a new concept using the query tool.
    Useful for explaining concepts to the user.
    Queries the database for relevant information,
    and provides it back in a structured manner


    """
    explanation = explain(query)
    console.print("\n[bold green]Explanation:[/bold green]\n")
    typer.echo(explanation + "\n")
    return explanation


query_db_tool = FunctionTool.from_defaults(fn=query)

test_example_tool = FunctionTool.from_defaults(fn=make_example)

find_test_tool = FunctionTool.from_defaults(fn=find_best_test)

explain_tool = FunctionTool.from_defaults(fn=explain_this)


statistics_agent = ReActAgent.from_tools(
    [query_db_tool, test_example_tool, find_test_tool, explain_tool], llm=COHERE_MODEL
)


@app.command()
def ask(query: str):
    """Launch an agentic query."""
    console = Console()
    console.print(f"\n[bold blue]Query:[/bold blue] {query}\n")

    response = statistics_agent.chat(query)

    console.print("\n[bold green]Final Response:[/bold green]")
    console.print(Markdown(response.response))


if __name__ == "__main__":
    app()

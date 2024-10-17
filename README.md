# Hypothesis Sage

Hypothesis Sage is a powerful command-line interface (CLI) tool designed to assist with statistical analysis and hypothesis testing. It leverages advanced language models and a curated database of statistical information to provide intelligent recommendations and explanations for various statistical scenarios.

We use Typer to create a simple CLI to access each tool, in addition to an agent that can answer complex questions.

In the future, we plan to create a web interface for access to Hypothesis Sage.

## Features

1. **Query Database**: Retrieve relevant statistical information from a curated database.
2. **Generate Test Examples**: Create detailed examples of how to apply specific statistical tests to given situations.
3. **Find Best Test**: Recommend appropriate statistical tests for a given scenario, including assumptions and their validity.
4. **Explain Concepts**: Provide explanations for statistical concepts and queries.
5. **AI-Powered Assistance**: Utilize an intelligent agent to answer complex statistical questions and provide comprehensive guidance.

## Installation


## Usage

Hypothesis Sage offers several commands to assist with your statistical analysis:

## Examples

Here are some example uses of Hypothesis Sage:

1. Query the database:
   ```
   python hsage_cli.py query "What is the difference between Type I and Type II errors?"
   ```

2. Generate a test example:
   ```
   python hsage_cli.py make-example "t-test" "Comparing mean heights of two groups"
   ```

3. Find the best test for a scenario:
   ```
   python hsage_cli.py find-best-test "I want to compare the effectiveness of three different teaching methods on student test scores"
   ```

4. Explain a statistical concept:
   ```
   python hsage_cli.py explain-this "What is the central limit theorem?"
   ```

5. Ask a complex question:
   ```
   python hsage_cli.py ask "How do I interpret the results of a multiple regression analysis with interaction terms?"
   ```

These commands will provide you with detailed information, examples, and guidance for your statistical analysis needs.

"""
ValueError: An output parsing error occurred. In order to pass this error back to the agent and have it try again, pass `handle_parsing_errors=True` to the AgentExecutor.
This is the error: Could not parse LLM output: `It seems like you are trying to sort a list of customers based on their last name and then their first name using
the `sort()` function in Python. Here is an example of how you can do this:
`
"""

import argparse

from kwwutils import clock, execute, get_llm, printit
from langchain.agents import load_tools
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool

customer_list = [
    ["Harrison", "Chase"],
    ["Lang", "Chain"],
    ["Dolly", "Too"],
    ["Elle", "Elem"],
    ["Geoff", "Fusion"],
    ["Trance", "Former"],
    ["Jen", "Ayai"],
]

question = """
Sort these customers by
last name and then first name
and print the output: {customer_list}
"""


@clock
@execute
def main(options):
    llm = get_llm(options)
    tools = load_tools(["llm-math", "wikipedia"], llm=llm)
    printit("tools", tools)

    agent = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        handle_parsing_errors=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    printit("agent", agent)
    response = agent.invoke({"input": question})
    printit("question", response)
    return response


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding",
        type=str,
        help="embedding: chroma gpt4all huggingface",
        default="gpt4all",
    )
    parser.add_argument(
        "--llm_type", type=str, help="llm_type: chat or llm", default="llm"
    )
    parser.add_argument("--repeatcnt", type=int, help="repeatcnt", default=1)
    parser.add_argument("--temperature", type=float, help="temperature", default=0.1)
    parser.add_argument("--model", type=str, help="model", default="openchat")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "codellama:7b",
            #       "llama2:latest",
            #       "medllama2:latest",
            #       "mistral:instruct",
            #       "mistrallite:latest",
            "openchat:latest",
            "orca-mini:latest",
            "vicuna:latest",
            #       "wizardcoder:latest",
        ],
    )
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    options = Options()
    main(**options)

"""
ValueError: An output parsing error occurred. In order to pass this error back to the agent and have it try again, pass `handle_parsing_errors=True` to the AgentExecutor.
This is the error: Could not parse LLM output: `It seems like you are trying to sort a list of customers based on their last name and then their first name using
the `sort()` function in Python. Here is an example of how you can do this:
`
"""

import argparse

import langchain
from kwwutils import clock, execute, get_llm, printit
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool

question = """
Generate a python script to query the YouTube API to retrieve horror movies.
First find the list of valid library we can use.
Use only opensource libraray so that we do not need to pay for the usage.
If no open source library is available then say so and stop.
Do you keep repeating the same step if you are not able to make progress. 
Repeat at most 2 times with the same step then say cannot make progress and quit.
Provide a list and the size of the movie files.
"""


@clock
@execute
def main(options):
    llm = get_llm(options)
    agent_executor = create_python_agent(
        llm,
        tool=PythonREPLTool(),
        handle_parsing_errors=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    printit("agent_executor", agent_executor)
    langchain.debug = True
    response = agent_executor.invoke({"input": question})
    langchain.debug = False
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
    parser.add_argument("--model", type=str, help="model", default="deepseek-coder")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "codebooga:latest",
            "codellama:13b",
            "codellama:13b-python",
            "codellama:34b",
            "codellama:34b-python",
            "codellama:7b",
            "codellama:7b-python",
            "codeup:latest",
            "deepseek-coder:latest",
            "dolphin-mistral:latest",
            "dolphin-mixtral:latest",
            "falcon:latest",
            "llama-pro:latest",
            "llama2-uncensored:latest",
            "llama2:latest",
            "magicoder:latest",
            "meditron:latest",
            "medllama2:latest",
            "mistral-openorca:latest",
            "mistral:instruct",
            "mistral:latest",
            "mistrallite:latest",
            "mixtral:latest",
            "openchat:latest",
            "orca-mini:latest",
            "orca2:latest",
            "phi:latest",
            "phind-codellama:latest",
            "sqlcoder:latest",
            "stable-code:latest",
            "starcoder:latest",
            "starling-lm:latest",
            "tinyllama:latest",
            "vicuna:latest",
            "wizardcoder:latest",
            "wizardlm-uncensored:latest",
            "yarn-llama2:latest",
            "yarn-mistral:latest",
        ],
    )
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    options = Options()
    main(**options)

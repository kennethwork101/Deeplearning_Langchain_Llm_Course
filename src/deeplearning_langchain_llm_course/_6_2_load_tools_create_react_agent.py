"""
/home/kenneth/learning/venv_latest/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:115: LangChainDeprecationWarning: The function `initialize_agent` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use Use new agent constructor methods like create_react_agent, create_json_agent, create_structured_chat_agent, etc. instead
"""

import argparse
from datetime import date

from kwwutils import clock, execute, get_llm, printit
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, tool
from langchain_community.agent_toolkits.load_tools import load_tools

questions = [
    "whats the date today?",
    "What is the 25% of 300?",
    """
    cual es su biografía? Por favor haz un resumen conciso de máximo 50 palabras en idioma español
    Gustavo Adolfo Becquer poeta y narrador español
    cual es su biografía? Por favor haz un resumen conciso de máximo 50 palabras en idioma español
    """,
    """
    Tom M. Mitchell is an American computer scientist and the Founders University Professor at Carnegie Mellon University (CMU)
    what book did he write?
    """,
]


''' 
@tool
def translate_into_english(self, string):
    """
    Use to translate from one language to another English
    """
    translater = Translator()
    out = translater.translate(string, dest='en')
    return out.text
'''


@tool
def timefn(text: str) -> str:
    """
    Returns todays date, use this for any questions related to knowing todays date.
    The input should always be an empty string, and this function will always return todays
    date - any date mathmatics should occur outside this function.
    """
    return str(date.today())


@clock
@execute
def main(options):
    llm = get_llm(options)
    tools = load_tools(["llm-math", "wikipedia"], llm=llm)
    tools.extend([timefn])
    #   tools.extend([translate_into_english])

    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=5,
        verbose=True,
    )
    printit("llm", llm)
    printit("tools", tools)
    printit("tools[0] type", type(tools[0]))
    printit("agent", agent)
    printit("agent type", type(agent))
    printit("agent_executor", agent_executor)
    printit("agent_executor type", type(agent_executor))
    responses = []
    for question in questions:
        response = agent_executor.invoke({"input": question})
        responses.append(response)
        printit("response", response)
    return responses


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
    parser.add_argument("--model", type=str, help="model", default="mistral")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "codellama:7b",
            "llama2:latest",
            "medllama2:latest",
            "mistral:instruct",
            "mistrallite:latest",
            "openchat:latest",
            "orca-mini:latest",
            "vicuna:latest",
            "wizardcoder:latest",
        ],
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    options = Options()
    main(**options)

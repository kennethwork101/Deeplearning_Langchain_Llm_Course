import argparse
from datetime import date

from kwwutils import clock, execute, get_llm, printit
from langchain.agents import AgentType, initialize_agent, tool
from langchain_community.agent_toolkits.load_tools import load_tools

questions = [
    "whats the date today?",
    "What is the 25% of 300?",
    """
        cual es su biografía? Por favor haz un resumen conciso de máximo 50 palabras en idioma español
        Gustavo Adolfo Becquer poeta y narrador español cual es su biografía? Por favor haz 
        un resumen conciso de máximo 50 palabras en idioma español
    """,
    """
        Tom M. Mitchell is an American computer scientist and the Founders University Professor at Carnegie Mellon University (CMU)
        what book did he write?
    """,
]


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
    tools = load_tools(tool_names=["llm-math", "wikipedia"], llm=llm)
    tools.extend([timefn])
    agent_executor = initialize_agent(
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True,
    )
    printit("llm", llm)
    printit("tools", tools)
    printit("agent_executor", agent_executor)
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
    parser.add_argument("--model", type=str, help="model", default="mistral:instruct")
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

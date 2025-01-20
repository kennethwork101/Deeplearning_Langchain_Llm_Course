"""
Note the outpt is different. It only has the overall summary when using LCEL.
Where as LLMChain output is a dict but LCEL is an AIMessage.
"""

import argparse
import os

import pandas as pd
from kwwutils import clock, execute, get_llm, printit
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

template1 = "Translate the following review to english: \n\n{Review}"
template2 = "Can you summarize the following review in 1 sentence: \n\n{English_Review}"
template3 = "What language is the following review:\n\n{Review}"
template4 = """
    Write a follow up response to the following
    summary in the specified language:
    \n\nSummary: {summary}\n\nLanguage: {language}
"""


@clock
@execute
def main(options):
    chat_llm = get_llm(options)

    # Build the file path so we can run via pytest from different directory and not impact where the file is located
    dirpath = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dirpath, "data.csv")
    df = pd.read_csv(file_path, delimiter=";")
    review = df.Review[3]
    printit("review", review)

    print(1)
    prompt1 = ChatPromptTemplate.from_template(template=template1)
    prompt2 = ChatPromptTemplate.from_template(template=template2)
    prompt3 = ChatPromptTemplate.from_template(template=template3)
    prompt4 = ChatPromptTemplate.from_template(template=template4)
    print(2)

    # Not sure why both works?
    chain_1 = prompt1 | chat_llm
    chain_2 = {"English_Review": chain_1} | prompt2 | chat_llm
    chain_3 = {"Review": chain_2} | prompt3 | chat_llm
    print(3)
    overall_chain = {"summary": chain_2, "language": chain_3} | prompt4 | chat_llm
    print(4)
    response = overall_chain.invoke({"Review": review})
    print(5)
    printit("overall response", response)

    # Not sure why both works as this one works too
    chain_1 = prompt1 | chat_llm
    chain_2 = {"English_Review": chain_1} | prompt2 | chat_llm
    chain_3 = {"Review": chain_2} | prompt3 | chat_llm
    overall_chain = (
        {"Review": RunnablePassthrough(), "summary": chain_2, "language": chain_3}
        | prompt4
        | chat_llm
    )
    response = overall_chain.invoke({"Review": review})
    printit("overall response", response)
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
        "--llm_type", type=str, help="llm_type: chat or llm", default="chat"
    )
    parser.add_argument("--repeatcnt", type=int, help="repeatcnt", default=1)
    parser.add_argument("--temperature", type=float, help="temperature", default=0.1)
    parser.add_argument("--model", type=str, help="model", default="llama2")
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
            "samantha-mistral:latest",
            "vicuna:latest",
            "wizardcoder:latest",
        ],
    )
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    options = Options()
    main(**options)

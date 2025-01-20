"""
LLMChain output is more detailed
dict_keys(['Review', 'English_Review', 'summary', 'language', 'followup_message']
"""

import argparse
import os

import pandas as pd
from kwwutils import clock, execute, get_llm, printit
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import ChatPromptTemplate

template1 = "Translate the following review to english: \n\n{Review}"
template2 = "Can you summarize the following review in 1 sentence: \n\n{English_Review}"
template3 = "What language is the following review:\n\n{Review}"
template4 = """
    Write a follow up response to the following summary in the specified language:
    \n\nSummary: {summary}\n\nLanguage: {language}
"""


@clock
@execute
def main(options):
    chat_llm = get_llm(options)
    # Build the file path so we can run via pytest from different directory and not impact where the file is located
    dirpath = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dirpath, "data.csv")
    printit("file_path", file_path)
    df = pd.read_csv(file_path, delimiter=";")
    review = df.Review[3]
    printit("review", review)

    prompt1 = ChatPromptTemplate.from_template(template=template1)
    prompt2 = ChatPromptTemplate.from_template(template=template2)
    prompt3 = ChatPromptTemplate.from_template(template=template3)
    prompt4 = ChatPromptTemplate.from_template(template=template4)

    chain1 = LLMChain(llm=chat_llm, prompt=prompt1, output_key="English_Review")
    chain2 = LLMChain(llm=chat_llm, prompt=prompt2, output_key="summary")
    chain3 = LLMChain(llm=chat_llm, prompt=prompt3, output_key="language")
    chain4 = LLMChain(llm=chat_llm, prompt=prompt4, output_key="followup_message")

    # Note cannot use simple sequential chain because template4 has more than 1 input
    overall_chain = SequentialChain(
        chains=[chain1, chain2, chain3, chain4],
        input_variables=["Review"],
        output_variables=["English_Review", "summary", "language", "followup_message"],
    )

    response = overall_chain.invoke({"Review": review})
    printit("1 overall response", response)

    response = overall_chain.invoke(review)
    printit("2 overall response", response)
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

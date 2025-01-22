import sys

_path = "../../../"


import argparse

import pandas as pd
from kwwutils import clock, execute, get_llm, printit
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser


@clock
@execute
def main(options):
    llm = get_llm(options)
    df = pd.read_csv("data.csv", delimiter=";")
    print(df.head())
    review = df.Review[3]
    print(f"\n>>>review::: \n{review}\n<<<")

    template1 = "Translate the following review to english: \n\n{Review}"
    prompt1 = ChatPromptTemplate.from_template(template=template1)
    chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="English_Review")

    template2 = (
        "Can you summarize the following review in 1 sentence: \n\n{English_Review}"
    )
    prompt2 = ChatPromptTemplate.from_template(template=template2)
    chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="summary")

    template3 = "What language is the following review:\n\n{Review}"
    prompt3 = ChatPromptTemplate.from_template(template=template3)
    chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="language")

    template4 = """
        Write a follow up response to the following
        summary in the specified language:
        \n\nSummary: {summary}\n\nLanguage: {language}
    """
    prompt4 = ChatPromptTemplate.from_template(template=template4)
    chain4 = LLMChain(llm=llm, prompt=prompt4, output_key="followup_message")

    overall_chain = SequentialChain(
        chains=[chain1, chain2, chain3, chain4],
        input_variables=["Review"],
        output_variables=["English_Review", "summary", "followup_message"],
        verbose=True,
    )
    response = overall_chain(review)
    printit(overall_chain, response)


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
    parser.add_argument("--model", type=str, help="model")
    """
    parser.add_argument('--model', type=str, help='model', default="llama2")
    """
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "codellama:7b",
            "everythinglm:latest",
            "falcon:latest",
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
    args = parser.parse_args()
    args = vars(args)
    return args


if __name__ == "__main__":
    options = Options()
    main(**options)

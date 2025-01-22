import sys

_path = "../../../"


import argparse

import pandas as pd
from kwwutils import clock, execute, get_llm, printit
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser


@clock
@execute
def main(options):
    llm = get_llm(options)
    product = "Amazon Kindle"
    printit("llm", llm)

    template1 = "What is the best name to describe a company that makes {product}?"
    prompt1 = ChatPromptTemplate.from_template(template=template1)
    chain1 = prompt1 | llm | StrOutputParser()

    template2 = "Write a 50 words description for the following company:{company_name}"
    prompt2 = ChatPromptTemplate.from_template(template=template2)
    chain2 = {"company_name": chain1} | prompt2 | llm | StrOutputParser()
    response = chain2.invoke({"product": product})
    printit(prompt2, response)


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
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument("--model", type=str, help="model", default="llama2")
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

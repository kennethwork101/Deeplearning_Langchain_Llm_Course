import argparse
import sys

import pandas as pd
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOllama
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


def get_llm(options):
    if options['llm_type'] == "llm":
        llm = Ollama(model=options['model'], temperature=options['temperature'], callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    elif options['llm_type'] == "chat":
        llm = ChatOllama(model=options['model'], temperature=options['temperature'], callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    return llm


def printit(key, data):
    try:
        print(f"\n{'#'*80}\nkey: >>>{key}<<<:\ndata: >>>>>>{str(data)}<<<<<<\nlen: {len(data)}\n{'#-' * 40}\n")
    except:
        print(f"\n{'#'*80}\nkey: >>>{key}<<<:\ndata: >>>>>>{str(data)}<<<<<<\n{'#-' * 40}\n")


def main(options):
    llm = get_llm(options)
    product = "Amazon Kindle"
    printit("llm", llm)

    template1 = "What is the best name to describe a company that makes {product}?"
    prompt1 = ChatPromptTemplate.from_template(template=template1)
    template2 = "Write a 50 words description for the following company:{company_name}"
    prompt2 = ChatPromptTemplate.from_template(template=template2)

    if options['chain_type'] == "llmchain":
        chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="company_name")
        chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="Review")
        response = chain2.run(product)
    elif options['chain_type'] == "lcel":
        chain1 = prompt1 | llm | StrOutputParser()
        chain2 = {"company_name": chain1} | prompt2 | llm | StrOutputParser()
        response = chain2.invoke({"product": product})
    printit(prompt2, response)
    

def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain_type', type=str, help='chain_type: lcel or llmchain', default='lcel')
    parser.add_argument('--llm_type', type=str, help='llm_type: chat or llm', default='chat')
    parser.add_argument('--model', type=str, help='model', default="llama2")
    parser.add_argument('--temperature', type=float, help='temperature', default=0.1)
    args = parser.parse_args()
    args = vars(args)
    return args


if __name__ == '__main__':
    options = Options()
    main(options)
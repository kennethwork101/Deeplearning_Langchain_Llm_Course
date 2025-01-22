import argparse

from kwwutils import clock, execute, get_llm, printit
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

template1 = "What is the best name to describe a company that makes {product}?"
template2 = "Write a 50 words description for the following company: {company_name}"


@clock
@execute
def main(options):
    chat_llm = get_llm(options)
    product = options["product"]
    output_parser = StrOutputParser()
    printit("llm", chat_llm)

    prompt1 = ChatPromptTemplate.from_template(template=template1)
    prompt2 = ChatPromptTemplate.from_template(template=template2)

    chain1 = LLMChain(llm=chat_llm, prompt=prompt1)
    chain2 = LLMChain(llm=chat_llm, prompt=prompt2)

    # Note all chains only has 1 input as required by SimpleSequentialChain
    overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

    response = overall_chain.invoke({"input": product})

    printit("prompt1", prompt1)
    printit("prompt2", prompt2)
    printit("chain1", chain1)
    printit("chain2", chain2)
    printit(f"llmchain overall_chain prompt2 {prompt2}", response)

    chain1 = prompt1 | chat_llm | output_parser
    chain2 = {"company_name": chain1} | prompt2 | chat_llm | output_parser
    response = chain2.invoke({"product": product})
    printit(f"lcel chain prompt2 {prompt2}", response)
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
    parser.add_argument("--product", type=str, help="product", default="Amazon Kindle")
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
    args = parser.parse_args()
    args = vars(args)
    return args


if __name__ == "__main__":
    options = Options()
    main(**options)

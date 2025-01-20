import argparse

from kwwutils import clock, execute, get_llm, get_vectordb, printit
from langchain.chains import RetrievalQA

_path = "../../"


@clock
@execute
def main(options):
    @clock
    def run(chain_type):
        response = qa.invoke({"query": question})
        printit(f"{chain_type}: query", response)
        return response

    llm = get_llm(options)
    vectordb = get_vectordb(options)
    retriever = vectordb.as_retriever()
    question = options["question"]
    chain_type = options["chain_type"]
    printit("chain_type", chain_type)

    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type=chain_type, retriever=retriever, verbose=True
    )
    printit(f"chai_type {chain_type} qa type:", type(qa))
    response = run(chain_type=chain_type)
    return response


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding", type=str, help="embedding: chroma gpt4all", default="chroma"
    )
    parser.add_argument(
        "--embedmodel", type=str, help="embedding: ", default="all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--llm_type", type=str, help="llm_type: chat or llm", default="llm"
    )
    parser.add_argument("--chain_type", type=str, help="chain_type", default="stuff")
    parser.add_argument(
        "--question",
        type=str,
        help="question",
        default="Please list all your shirts with sun protection in a table in markdown and summarize each one but make it short",
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        help="persist_directory",
        default=f"{_path}mydb/data_all/",
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
            #       "mistral:instruct",
            "mistrallite:latest",
            #       "openchat:latest",
            "orca-mini:latest",
            "vicuna:latest",
            "wizardcoder:latest",
        ],
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    options = Options()
    main(**options)

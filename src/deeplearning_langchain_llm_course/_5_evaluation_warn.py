"""
/home/kenneth/learning/venv_latest/lib/python3.10/site-packages/langchain/chains/llm.py:344: UserWarning: The apply_and_parse method is deprecated, instead pass an output parser directly to LLMChain.
/home/kenneth/learning/venv_latest/lib/python3.10/site-packages/langchain_core/_api/deprecation.py:115: LangChainDeprecationWarning: The function `apply` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use batch instead.
"""

import argparse

from kwwutils import clock, execute, get_llm, get_vectordb, printit
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain, QAGenerateChain
from langchain_community.document_loaders import CSVLoader

_path = "../../"


@clock
@execute
def main(options):
    llm = get_llm(options)
    vectordb = get_vectordb(options)
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, verbose=True
    )
    printit("qa", qa)
    printit("qa type", type(qa))

    loader = CSVLoader(file_path=options["filename"])
    data = loader.load()
    test_size = options["test_size"]
    print(f"test_size: {test_size}")
    datalen = len(data)

    example_gen_chain = QAGenerateChain.from_llm(llm)
    printit("example_gen_chain", example_gen_chain)

    new_examples = example_gen_chain.apply_and_parse(
        [{"doc": t} for t in data[:test_size]]
    )
    examples = [q["qa_pairs"] for q in new_examples]
    printit("new_examples", new_examples)

    responses = []
    for i, q in enumerate(examples):
        response = qa.invoke({"query": q["query"]})
        responses.append(response)
        printit(f"i: {i} querry: {q['query']}", response)
        if i >= options["test_size"]:
            break
    print(f"new_examples len {len(new_examples)}")
    print(f"datalen {datalen}")
    test_examples = examples[:test_size]

    predictions = qa.apply(test_examples)
    eval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = eval_chain.evaluate(test_examples, predictions)

    for i, eg in enumerate(test_examples):
        print(f"Example {i}: {eg}")
        print("Question: " + predictions[i]["query"])
        print("Real Answer: " + predictions[i]["answer"])
        print("Predicted Answer: " + predictions[i]["result"])
        print("Predicted Grade: " + graded_outputs[i]["results"])
        print()
    return responses


def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chain_type",
        type=str,
        help="chain_type: stuff, map_reduce, map_rerank, refine",
        default="stuff",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        help="embedding: chroma gpt4all huggingface",
        default="gpt4all",
    )
    parser.add_argument(
        "--embedmodel", type=str, help="embedding: ", default="all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="filename",
        default="../../data/data_all/csv_files/OutdoorClothingCatalog_1000.csv",
    )
    parser.add_argument(
        "--llm_type", type=str, help="llm_type: chat or llm", default="llm"
    )
    parser.add_argument(
        "--persist_directory",
        type=str,
        help="persist_directory",
        default=f"{_path}mydb/data_all/",
    )
    parser.add_argument("--repeatcnt", type=int, help="repeatcnt", default=1)
    parser.add_argument("--test_size", type=int, help="test_size", default=10)
    parser.add_argument("--temperature", type=float, help="temperature", default=0.1)
    parser.add_argument("--model", type=str, help="model", default="vicuna")
    """
    parser.add_argument('--model', type=str, help='model')
    """
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "codellama:7b",
            "llama2:latest",
            #       "medllama2:latest",
            "mistral:instruct",
            #       "mistrallite:latest",
            "openchat:latest",
            "orca-mini:latest",
            "vicuna:latest",
            #       "wizardcoder:latest",
        ],
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    options = Options()
    main(**options)

"""
memory.buffer is not working correctly as it is empty.
"""

import argparse

from kwwutils import clock, execute, get_llm, printit
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

schedule = """
There is a meeting at 8am with your product team.  You will need your powerpoint presentation prepared.
9am-12pm have time to work on your LangChain project which will go quickly because Langchain is such a powerful tool.
At Noon, lunch at the italian resturant with a customer who is driving from over an hour away to meet you to 
understand the latest in AI.  Be sure to bring your laptop to show the latest LLM demo.
"""


@clock
@execute
def main(options):
    llm = get_llm(options)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    conversation.invoke(input="Hi, my name is Kenneth")
    conversation.invoke(input="What is 1+1?")
    conversation.invoke(input="What is my name?")
    conversation.invoke(
        input=" I have given you my name earlier. Can you repeat it back to me? What is my name?"
    )
    printit("1111 ConversationChain memory.buffer", memory.buffer)

    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
    memory.save_context({"input": "Hello"}, {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"}, {"output": "Cool"})
    memory.save_context(
        {
            "input": "What is on the schedule today? Do not make things up but answer with what you know"
        },
        {"output": f"{schedule}"},
    )
    memory.load_memory_variables({})
    printit("2222 ConversationSummaryBufferMemory memory.buffer", memory.buffer)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    # response = conversation.invoke(input="What is 1+2?")
    response = conversation.invoke(
        input="What is 1+2 and report the result as an integer?"
    )
    printit("3333 ConversationSummaryBufferMemory memory.buffer", memory.buffer)
    printit("response", response)
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
        "--embedmodel", type=str, help="embedding: ", default="all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--llm_type", type=str, help="llm_type: chat or llm", default="chat"
    )
    parser.add_argument("--repeatcnt", type=int, help="repeatcnt", default=1)
    parser.add_argument("--temperature", type=float, help="temperature", default=0.1)
    """
    parser.add_argument('--model', type=str, help='model', default="llama2")
    """
    parser.add_argument("--model", type=str, help="model")
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
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    options = Options()
    print(f"options: {options}")
    main(**options)

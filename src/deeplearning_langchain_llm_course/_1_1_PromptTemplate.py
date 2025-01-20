import argparse

from kwwutils import clock, execute, get_llm, printit
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

customer_email = """
Arrr, I be fuming that me blender lid
flew off and splattered me kitchen walls
with smoothie! And to make matters worse,
the warranty don't cover the cost of
cleaning up me kitchen. I need yer help
right now, matey!
"""

style = "American English in a calm and respectful tone"

customer_template = """
Translate the text
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

service_reply = """
Hey there customer, the warranty does not cover cleaning expenses for your kitchen
because it's your fault that you misused your blender
by forgetting to put the lid on before starting the blender.
Tough luck! See ya!
"""

service_style_pirate = "a polite tone that speaks in Spanish"

service_template = """
Translate the text
that is delimited by triple backticks 
into a style that is {style}.
text: ```{service_reply}```
"""


@clock
@execute
def main(options):
    llm = get_llm(options)

    customer_prompt = PromptTemplate(
        input_variables=["style", "customer_email"], template=customer_template
    )
    service_prompt = PromptTemplate(
        input_variables=["style", "service_reply"], template=service_template
    )

    chain1 = LLMChain(llm=llm, prompt=customer_prompt)
    chain2 = LLMChain(llm=llm, prompt=service_prompt)

    response1 = chain1.invoke({"style": style, "customer_email": customer_email})
    response2 = chain2.invoke(
        {"style": service_style_pirate, "service_reply": service_reply}
    )

    printit(f"customer_prompt {customer_prompt}", response1)
    printit(f"service_prompt {service_prompt}", response2)
    printit("chain1", chain1)
    printit("chain2", chain2)
    printit("response1", response1)
    printit("response2", response2)
    return response1, response2


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
            "vicuna:latest",
            "wizardcoder:latest",
        ],
    )
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    options = Options()
    main(**options)

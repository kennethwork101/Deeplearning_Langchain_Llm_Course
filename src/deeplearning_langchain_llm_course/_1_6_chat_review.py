import argparse

from kwwutils import clock, execute, get_llm, printit
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate

customer_review = """
This leaf blower is pretty amazing.  It has four settings: candle blower, gentle breeze, windy city, and tornado.
It arrived in two days, just in time for my wife's anniversary present.
I think my wife liked it so much she was speechless.  So far I've been the only one using it, and I've been
using it every other morning to clear the leaves on our lawn.  It's slightly more expensive than the other leaf blowers
out there, but I think it's worth it for the extra features.
"""

review_template = """
For the following text, extract the following information: gift: Was the item purchased as a gift for someone else?
Answer True if yes, False if not or unknown.  delivery_days: How many days did it take for the product
to arrive? If this information is not found, output -1.  price_value: Extract any sentences about the value or price,
and output them as a comma separated Python list.
text: {text}
{format_instructions}
    """


@clock
@execute
def main(options):
    chat_llm = get_llm(options)

    gift_schema = ResponseSchema(
        name="gift",
        description="Was the item purchased as a gift for someone else?  Answer True if yes, False if not or unknown.",
    )
    delivery_days_schema = ResponseSchema(
        name="delivery_days",
        description="How many days did it take in whole number for the product to arrive? If this information is not found, output -1.",
    )
    price_value_schema = ResponseSchema(
        name="price_value",
        description="Extract any sentences about the value or price, and output them as a comma separated Python list.",
    )

    response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    review_prompt = ChatPromptTemplate.from_template(template=review_template)

    chain = review_prompt | chat_llm
    response = chain.invoke(
        {"text": customer_review, "format_instructions": format_instructions}
    )

    printit("response_schemas", response_schemas)
    printit("output_parser", output_parser)
    printit("format_instructions", format_instructions)
    printit("chat response.content", response.content)
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
            "vicuna:latest",
            "wizardcoder:latest",
        ],
    )
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    options = Options()
    main(**options)

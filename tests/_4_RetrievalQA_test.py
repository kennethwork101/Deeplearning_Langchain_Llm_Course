import pytest
from kwwutils import clock, printit

from uvprog2025.Deeplearning_Langchain_Llm_Course.src.deeplearning_langchain_llm_course._4_RetrievalQA import (
    main,
)


@clock
# @pytest.mark.parametrize("chain_type", ["stuff", "map_reduce", "map_rerank", "refine"])
# @pytest.mark.parametrize("chain_type", ["stuff", "map_reduce"])
@pytest.mark.parametrize("chain_type", ["stuff"])
def test_func(options, model, chain_type):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["chain_type"] = chain_type
    options["llm_type"] = "llm"
    options["question"] = (
        "Please list all your shirts with sun protection in a table in markdown and summarize each one but make it short'"
    )
    response = main(**options)
    printit("response", response)
    result = response["result"].lower()
    assert "shirt" in result

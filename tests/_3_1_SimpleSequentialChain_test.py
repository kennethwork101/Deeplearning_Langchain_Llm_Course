import pytest
from kwwutils import clock, printit

from uvprog2025.Deeplearning_Langchain_Llm_Course.src.deeplearning_langchain_llm_course._3_1_SimpleSequentialChain import (
    main,
)


@pytest.mark.testme2
@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["product"] = "Amazon"
    response = main(**options)
    printit("response", response)
    assert "Amazon" in response

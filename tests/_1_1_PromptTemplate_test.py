import pytest
from kwwutils import clock, printit

from uvprog2025.Deeplearning_Langchain_Llm_Course.src.deeplearning_langchain_llm_course._1_1_PromptTemplate import (
    main,
)


@pytest.mark.testme
@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "llm"
    response1, response2 = main(**options)
    printit("response1", response1)
    printit("response2", response2)
    assert sorted(response1.keys()) == ["customer_email", "style", "text"]
    assert sorted(response2.keys()) == ["service_reply", "style", "text"]

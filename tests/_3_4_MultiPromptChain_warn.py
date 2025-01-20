from kwwutils import clock, printit

from uvprog2025.Deeplearning_Langchain_Llm_Course.src.deeplearning_langchain_llm_course._3_4_MultiPromptChain_warn import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["question"] = "What is black body radiation?"
    response = main(**options)
    printit("response", response)
    result = response["text"].lower()
    assert "black" in result
    assert "body radiation" in result
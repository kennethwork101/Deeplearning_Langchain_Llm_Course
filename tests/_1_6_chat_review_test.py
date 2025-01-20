from kwwutils import clock, printit

# from uvprog2025.Deeplearning_Langchain_Llm_Course.src.deeplearning_langchain_llm_course._1_6_chat_review import (
from uvprog2025.Deeplearning_Langchain_Llm_Course.src.deeplearning_langchain_llm_course._1_6_chat_review import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    response = main(**options)
    printit("response", response)
    assert "gift" in response.content
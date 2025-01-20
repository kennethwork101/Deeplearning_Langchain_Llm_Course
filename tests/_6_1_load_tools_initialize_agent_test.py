from kwwutils import clock, printit

from uvprog2025.Deeplearning_Langchain_Llm_Course.src.deeplearning_langchain_llm_course._6_1_load_tools_initialize_agent import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "llm"
    options["question"] = "What did they say about fossil fuel projects?"
    response = main(**options)
    printit("response", response)
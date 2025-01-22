from kwwutils import clock, printit

from uvprog2025.Deeplearning_Langchain_Llm_Course.src.deeplearning_langchain_llm_course._6_2_load_tools_create_react_agent import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    #   options["model"] = "llama2"
    options["llm_type"] = "llm"
    responses = main(**options)
    printit("responses", responses)
    for response in responses:
        assert sorted(response.keys()) == ["input", "output"]

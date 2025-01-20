from kwwutils import clock, printit

from uvprog2025.Deeplearning_Langchain_Llm_Course.src.deeplearning_langchain_llm_course._6_3_create_python_agent_bad import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "llm"
    response = main(**options)
    printit("response", response)
    assert sorted(response.keys()) == ["input", "output"]
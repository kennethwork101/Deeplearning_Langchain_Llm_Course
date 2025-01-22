from kwwutils import clock, printit

from uvprog2025.Deeplearning_Langchain_Llm_Course.src.deeplearning_langchain_llm_course._1_4_ResponseSchema import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    response1, response2, response3 = main(**options)
    printit("response1", response1)
    printit("response2", response2)
    printit("response3", response3)
    assert response1
    assert response2
    assert response3

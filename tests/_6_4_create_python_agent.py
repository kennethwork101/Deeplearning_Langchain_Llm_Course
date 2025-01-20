from kwwutils import clock, printit

from uvprog2025.Deeplearning_Langchain_Llm_Course.src.deeplearning_langchain_llm_course._6_4_create_python_agent import (
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
    assert response
    assert sorted(response.keys()) == ["input", "output"]
    printit("response output", response["output"])
    """ 
    output = json.loads(response['output'])
    printit("output", output)
    expected_value = [{'first_name': 'Alice', 'last_name': 'Brown'}, 
                      {'first_name': 'John', 'last_name': 'Doe'}, 
                      {'first_name': 'Jane', 'last_name': 'Smith'}
                    ]
    """
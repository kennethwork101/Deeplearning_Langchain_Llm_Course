import os

from kwwutils import clock, printit

from uvprog2025.Deeplearning_Langchain_Llm_Course.src.deeplearning_langchain_llm_course._5_evaluation_warn import (
    main,
)


@clock
def test_func(pytestconfig, options, model):
    package_root = pytestconfig.rootpath
    filepath = "data/data_all/csv_files/OutdoorClothingCatalog_1000.csv"
    filename = os.path.join(package_root, filepath)
    printit("test_func filename", filename)
    printit("test_func model", model)
    options["filename"] = filename
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "llm"
    options["test_size"] = 10
    responses = main(**options)
    printit("responses", responses)
    assert responses
    for response in responses:
        printit("response", response)
        assert sorted(response.keys()) == ["query", "result"]
from project.data.datasets.ARG_KP_2021.processor import ArgKP21Processor
from project.data.datasets.ARG_KP_2021.task1_evaluator import ArgKP21Task1Evaluator
from project.data.datasets.processor_base import BaseDatasetProcessor
from project.data.datasets.validator_base import BaseDatasetEvaluator

processors_dict: dict = {"ARG_KP_2021_processor": ArgKP21Processor}

validators_dict: dict = {"ARG_KP_2021_validator": ArgKP21Task1Evaluator}


def dataset_processor_factory(name: str, params: dict = None) -> BaseDatasetProcessor:
    params = params if params else {}
    if name in processors_dict.keys():
        return processors_dict[name](**params)
    else:
        raise ValueError("{} is not a valid dataset processor type".format(name))


def dataset_validator_factory(name: str, params: dict = None) -> BaseDatasetEvaluator:
    params = params if params else {}
    if name in validators_dict.keys():
        return validators_dict[name](**params)
    else:
        raise ValueError("{} is not a valid dataset validator type".format(name))

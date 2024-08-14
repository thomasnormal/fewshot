import dotenv
import instructor
import openai
from pydantic import BaseModel, Field
import pytest

from fewshot.utils import signature


def test_from_string():
    assert signature("out1, out2").model_json_schema() == {'properties': {'out1': {'title': 'Out1', 'type': 'string'}, 'out2': {'title': 'Out2', 'type': 'string'}}, 'required': ['out1', 'out2'], 'title': 'Answer', 'type': 'object'}

def test_typed():
    assert signature("out1:list[str], out2:int").model_json_schema() == {'properties': {'out1': {'items': {'type': 'string'}, 'title': 'Out1', 'type': 'array'}, 'out2': {'title': 'Out2', 'type': 'integer'}}, 'required': ['out1', 'out2'], 'title': 'Answer', 'type': 'object'}


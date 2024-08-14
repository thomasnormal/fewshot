from pydantic import BaseModel
from examples.image_datasets import generate_image_with_circles
from fewshot.templates import Image, format_input_simple, gpt_format_image, PILImage


def test_custom_image_template():
    class MyImage(Image):
        def __init__(self, data: str):
            super().__init__()
            self._data = data

        def base64(self):
            return self._data

    class MyModel(BaseModel):
        img: MyImage
        other_data: str
        more_images: list[MyImage]

    model_instance = MyModel(
        img=MyImage(data="image_data_1"),
        other_data="some other data",
        more_images=[MyImage(data="image_data_2"), MyImage(data="image_data_3")],
    )

    formatted = format_input_simple(model_instance, gpt_format_image)
    assert formatted["content"][0]["type"] == "text"
    for i in range(1, 7):
        if i % 2 == 1:
            assert formatted["content"][i]["type"] == "text"
            assert formatted["content"][i]["text"] == f"[image {i//2 + 1}]:"
        else:
            assert formatted["content"][i]["type"] in ("image", "image_url")

    assert MyModel.model_json_schema() == {
        "$defs": {"MyImage": {"properties": {}, "title": "MyImage", "type": "object"}},
        "properties": {
            "img": {"$ref": "#/$defs/MyImage"},
            "other_data": {"title": "Other Data", "type": "string"},
            "more_images": {
                "items": {"$ref": "#/$defs/MyImage"},
                "title": "More Images",
                "type": "array",
            },
        },
        "required": ["img", "other_data", "more_images"],
        "title": "MyModel",
        "type": "object",
    }


def test_pil_image():
    class MyModel(BaseModel):
        other_data: str
        images: list[PILImage]

    # Make some PIL images
    model_instance = MyModel(
        other_data="...",
        images=[PILImage(generate_image_with_circles()[0]) for _ in range(2)],
    )
    formatted = format_input_simple(model_instance, gpt_format_image)
    assert len(formatted["content"]) == 5

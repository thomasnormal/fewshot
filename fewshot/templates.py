import base64
from io import BytesIO
import json
from typing import Annotated, Any, Callable

from pydantic import BaseModel


Base64Image = Annotated[str, "base64image"]


def image_to_base64(image):
    # image = image.resize((100, 100))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64_image_field(model: type[BaseModel], field_name: str) -> bool:
    field = model.model_fields[field_name]
    return "base64image" in field.metadata


def make_title(model) -> str:
    return f"Title-{model.__name__}"


# Define the custom ImageType class
class Image(BaseModel):
    def base64(self):
        raise NotImplementedError


class PILImage(Image):
    def __init__(self, image, media_type="image/png"):
        super().__init__()
        self._image = image
        self._media_type = media_type

    def base64(self):
        buffered = BytesIO()
        self._image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


# Helper function to recursively apply a function over any type of structure
def map_images(obj: Any, transform_func: Any) -> Any:
    if isinstance(obj, Image):
        return transform_func(obj)
    if isinstance(obj, dict):
        return {k: map_images(v, transform_func) for k, v in obj.items()}
    if isinstance(obj, list):
        return [map_images(elem, transform_func) for elem in obj]
    if isinstance(obj, BaseModel):
        return {k: map_images(getattr(obj, k), transform_func) for k in obj.model_fields}
    return obj


def format_input_simple(pydantic_object: BaseModel, img_formatter=None) -> dict[str, Any]:
    if img_formatter is None:
        img_formatter = gpt_format_image

    image_map = {}

    def replace_image_with_id(obj: Any) -> Any:
        image_id = f"[image {len(image_map) + 1}]"
        image_map[image_id] = obj.base64()
        return image_id

    dict_obj = map_images(pydantic_object, replace_image_with_id)
    processed = json.dumps(dict_obj)

    content = [{"type": "text", "text": processed}]
    for image_id, image in image_map.items():
        content.append({"type": "text", "text": image_id + ":"})
        content.append(img_formatter(image))

    return {"role": "user", "content": content}


def pydantic_template(
    pydantic_object: BaseModel, img_formatter: Callable[[str], dict[str, str]]
) -> dict[str, str]:
    content = []
    schema = pydantic_object.model_json_schema()
    properties = schema.get("properties", {})
    for field_name, field_info in pydantic_object.model_fields.items():
        if not field_info.exclude:
            value = getattr(pydantic_object, field_name)
            schema_info = properties.get(field_name, {})
            title = schema_info.get("title", field_name.title())
            content.append({"type": "text", "text": f"{title}:"})
            if is_base64_image_field(type(pydantic_object), field_name):
                content.append(img_formatter(value))
            elif hasattr(value, "model_dump_json"):
                content.append({"type": "text", "text": value.model_dump_json()})
            else:
                content.append({"type": "text", "text": str(value)})

    return {"role": "user", "content": content}


def gpt_format_image(base64: str, type: str = "base64", media_type: str = "image/png") -> dict[str, str]:
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media_type};{type},{base64}",
            # https://platform.openai.com/docs/guides/vision/low-or-high-fidelity-image-understanding
            # low will enable the "low res" mode. The model will receive a low-res 512px x 512px version of the image, and represent the image with a budget of 85 tokens. This allows the API to return faster responses and consume fewer input tokens for use cases that do not require high detail.
            # high will enable "high res" mode, which first allows the model to first see the low res image (using 85 tokens) and then creates detailed crops using 170 tokens for each 512px x 512px tile.
            "detail": "high",
        },
    }


def claude_format_image(base64: str, type: str = "base64", media_type: str = "image/png") -> dict[str, str]:
    return {
        "type": "image",
        "source": {
            "type": type,
            "media_type": media_type,
            "data": base64,
        },
    }


def format_input(pydantic_object: BaseModel) -> dict[str, str]:
    return pydantic_template(pydantic_object, gpt_format_image)


def format_input_claude(pydantic_object: BaseModel) -> dict[str, str]:
    return pydantic_template(pydantic_object, claude_format_image)

import base64
from io import BytesIO
from pydantic import BaseModel
from typing import Annotated, Callable

Base64Image = Annotated[str, "base64image"]


def image_to_base64(image):
    # image = image.resize((100, 100))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64_image_field(model: type[BaseModel], field_name: str) -> bool:
    field = model.model_fields[field_name]
    return "base64image" in field.metadata


def format_input_simple(pydantic_object: BaseModel) -> dict[str, str]:
    # Note: Doesn't support images
    yield {"role": "user", "content": pydantic_object.model_dump_json()}


def pydantic_template(
    pydantic_object: BaseModel, img_formatter: Callable[str, dict[str, str]]
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


def format_input(pydantic_object: BaseModel) -> dict[str, str]:
    return pydantic_template(
        pydantic_object,
        lambda base64: {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64}",
            },
        },
    )


def format_input_claude(pydantic_object: BaseModel) -> dict[str, str]:
    return pydantic_template(
        pydantic_object,
        lambda base64: {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64,
            },
        },
    )

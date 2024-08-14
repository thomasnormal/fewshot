import asyncio
import aiohttp
from pydantic import BaseModel
from typing import Any, Type, TypeVar, get_type_hints

T = TypeVar("T", bound=BaseModel)


class DeferredModel:
    def __init__(
        self,
        model_cls: Type[T],
        data_future: asyncio.Future,
        field_name: str = None,
        parent: "DeferredModel" = None,
    ):
        self._model_cls = model_cls
        self._data_future = data_future
        self._field_name = field_name
        self._parent = parent
        self._instance = None
        self._lock = asyncio.Lock()

    async def unwrap(self) -> Any:
        async with self._lock:
            if self._instance is None:
                if self._parent is None:
                    # Root model: fetch and parse data
                    data = await self._data_future
                    self._instance = self._model_cls.parse_obj(data)
                else:
                    # Nested field: wait for the parent to resolve and get the field value
                    parent_instance = await self._parent.unwrap()
                    self._instance = getattr(parent_instance, self._field_name)
        return self._instance

    def __getattr__(self, name: str) -> "DeferredModel":
        # For any attribute access, return a new DeferredModel representing that field
        return DeferredModel(
            self._model_cls, self._data_future, field_name=name, parent=self
        )


async def fetch_data(http_address: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(http_address) as response:
            return await response.json()


def wrap_pydantic_class(model_cls: Type[T], http_address: str) -> DeferredModel:
    data_future = asyncio.ensure_future(fetch_data(http_address))
    return DeferredModel(model_cls, data_future)


# Example Pydantic models
class Name(BaseModel):
    first: str
    last: str


class MyModel(BaseModel):
    name: Name
    number: int


# Usage example
async def main():
    wrapped_res = wrap_pydantic_class(MyModel, "http://example.com/api/data")

    # Other code can execute while the HTTP request is in progress

    # When needed, unwrap the values
    name = await wrapped_res.name.unwrap()  # Unwraps to a `Name` instance
    first_name = await wrapped_res.name.first.unwrap()  # Unwrap the first name
    last_name = await wrapped_res.name.last.unwrap()  # Unwrap the last name
    number = await wrapped_res.number.unwrap()  # Unwrap the number

    print(first_name, last_name, number)


# Run the example
if __name__ == "__main__":
    asyncio.run(main())

from typing import Any, Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: int = 200
    message: str = "ok"
    data: T | None = None

    @classmethod
    def success(cls, data: Any = None, message: str = "ok") -> "ApiResponse":
        return cls(code=200, message=message, data=data)

    @classmethod
    def error(cls, code: int = 500, message: str = "error", data: Any = None) -> "ApiResponse":
        return cls(code=code, message=message, data=data)


class PaginatedData(BaseModel):
    items: list[Any]
    total: int
    page: int
    page_size: int

from __future__ import annotations

import inspect
from typing import Awaitable, TypeVar, Union

T = TypeVar("T")

async def resolve_awaitable(value: Union[T, Awaitable[T]]) -> T:
    """Await value if needed and return the resolved result."""
    if inspect.isawaitable(value):
        return await value  # type: ignore[arg-type]
    return value  # type: ignore[return-value]

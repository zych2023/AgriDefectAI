from fastapi import Request
from fastapi.responses import JSONResponse
from app.common.response import ApiResponse


class BusinessException(Exception):
    def __init__(self, code: int = 400, message: str = "Bad Request"):
        self.code = code
        self.message = message


class UnauthorizedException(BusinessException):
    def __init__(self, message: str = "Unauthorized"):
        super().__init__(code=401, message=message)


class ForbiddenException(BusinessException):
    def __init__(self, message: str = "Forbidden"):
        super().__init__(code=403, message=message)


class NotFoundException(BusinessException):
    def __init__(self, message: str = "Not Found"):
        super().__init__(code=404, message=message)


async def business_exception_handler(request: Request, exc: BusinessException):
    return JSONResponse(
        status_code=exc.code,
        content=ApiResponse.error(code=exc.code, message=exc.message).model_dump(),
    )


async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=ApiResponse.error(code=500, message="Internal Server Error").model_dump(),
    )


def register_exception_handlers(app):
    app.add_exception_handler(BusinessException, business_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

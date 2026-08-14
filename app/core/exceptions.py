from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

class AppException(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message, status_code)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    message = first["msg"]

    return JSONResponse(
        status_code=422,
        content={"error": message}
    )

async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message}
    )
#


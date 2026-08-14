from starlette.responses import JSONResponse
from fastapi import status
from fastapi.encoders import jsonable_encoder
def success_response(message: str, data=None):
    return JSONResponse(status_code=status.HTTP_200_OK, content={"detail": message, "data": jsonable_encoder(data, exclude={"password"})})
def error_response(message: str, status_code: int):
    if status_code == 400:
        status_code = status.HTTP_400_BAD_REQUEST
    elif status_code == 401:
        status_code = status.HTTP_401_UNAUTHORIZED
    elif status_code == 403:
        status_code = status.HTTP_403_FORBIDDEN
    elif status_code == 404:
        status_code = status.HTTP_404_NOT_FOUND
    elif status_code == 405:
        status_code = status.HTTP_405_METHOD_NOT_ALLOWED
    return JSONResponse(status_code=status_code, content={"detail": message})
def register_response(data=None, token=None, token_type=None):
    return JSONResponse(status_code=status.HTTP_201_CREATED, content={
        "detail": "Supervisor account created successfully.",
        "data": jsonable_encoder(data, exclude={"password"}),
        "token": token,
        "token_type": token_type,
    }
)
def login_response(data=None, token=None, token_type=None):
    return JSONResponse(status_code=status.HTTP_200_OK, content={
        "detail":"Logged in done successfully.",
        "data": jsonable_encoder(data, exclude={"password"}),
        "token": token,
        "token_type": token_type
    }
)


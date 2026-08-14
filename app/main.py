from fastapi import FastAPI,Request
from fastapi.exceptions import RequestValidationError

from app.core.lifespan import lifespan
from app.api.routers import supervisor
from app.api.routers.supervisor import manage_system,manage_session,manage_student
from app.core.exceptions import validation_exception_handler, AppException, app_exception_handler

application = FastAPI(title="Smart Attendance", lifespan=lifespan)
application.add_exception_handler(RequestValidationError, validation_exception_handler)
application.add_exception_handler(AppException, app_exception_handler)

# routers
application.include_router(supervisor.router)
application.include_router(manage_system.router)
application.include_router(manage_session.router)
application.include_router(manage_student.router)

@application.get("/ai/health")
async def ai_health(request: Request):
    return request.app.state.ai.get_health()



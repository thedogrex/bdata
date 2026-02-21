import uvicorn
from predictor.api import app
from fastapi import Request, Response
from fastapi.middleware import Middleware
import logging

# Configuration: Set to False to enable /api/tasks/status logs, True to filter them out
FILTER_TASKS_STATUS_LOGS = True

class AccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not FILTER_TASKS_STATUS_LOGS:
            return True

        try:
            if hasattr(record, "args") and len(record.args) >= 3:
                path = record.args[2]
                if "/api/tasks/status" in path or "/favicon.ico" in path:
                    return False
        except Exception:
            pass

        return True

# Add custom middleware to filter access logs
@app.middleware("http")
async def filter_tasks_status_logs(request: Request, call_next):
    response = await call_next(request)
    
    # Skip logging for /api/tasks/status and /favicon.ico if enabled
    if FILTER_TASKS_STATUS_LOGS and (request.url.path == "/api/tasks/status" or request.url.path == "/favicon.ico"):
        return response
    
    return response

if __name__ == "__main__":
    # Configure uvicorn logger
    uvicorn_config = uvicorn.Config(
        app, 
        host="127.0.0.1", 
        port=8000,
        access_log=True
    )
    
    # Add filter to uvicorn's access logger if filtering is enabled
    if FILTER_TASKS_STATUS_LOGS:
        access_logger = logging.getLogger("uvicorn.access")
        access_logger.addFilter(AccessLogFilter())
    
    server = uvicorn.Server(uvicorn_config)
    server.run()

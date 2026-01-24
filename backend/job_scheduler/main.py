from fastapi import FastAPI
from job_scheduler.routes import jobs, health

app = FastAPI(
    title="Job Scheduler",
    version="1.0.0",
)

app.include_router(jobs.router)
app.include_router(health.router)


@app.get("/")
def read_root():
    return {
        "message": "Job Scheduler",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/generate/user/{user_id}",
            "status": "/api/status/user/{user_id}/{job_id}",
            "download": "/api/download/user/{user_id}/{job_id}",
        },
    }

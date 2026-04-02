from fastapi import FastAPI
from job_scheduler.routes import jobs, health, auth, training

app = FastAPI(
    title="Job Scheduler",
    version="1.0.0",
)

app.include_router(auth.router)
app.include_router(jobs.router)
app.include_router(health.router)
app.include_router(training.router)


@app.get("/")
def read_root():
    return {
        "message": "Job Scheduler",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/generate/user/{user_id}",
            "status": "/api/status/user/{user_id}/{job_id}",
            "download": "/api/download/user/{user_id}/{job_id}",
            "history": "/api/history/user/{user_id}",
        },
    }

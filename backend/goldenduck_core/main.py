from fastapi import FastAPI
from goldenduck_core.routes import jobs, health, auth, training, users

app = FastAPI(
    title="goldenduck-core",
    version="1.0.0",
)

app.include_router(auth.router)
app.include_router(jobs.router)
app.include_router(health.router)
app.include_router(training.router)
app.include_router(users.router)


@app.get("/")
def read_root():
    return {
        "message": "Job Scheduler",
        "version": "1.0.0",
        "endpoints": {
            "generate": "/api/generate/user/{user_id}",
            "status": "/api/status/user/{user_id}/{job_id}",
            "download": "/api/download/user/{user_id}/{job_id}",
            "download_selected_path": "/api/download/user/{user_id}/{job_id}/selected-path",
            "history": "/api/history/user/{user_id}",
        },
    }

"""GC-GARCH Synthetic Data Generator API (scaffold)."""
from fastapi import FastAPI

app = FastAPI(
    title="GC-GARCH Synthetic Data Generator",
    description="Generate synthetic OHLCV data using Gram-Charlier GARCH",
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {
        "message": "GC-GARCH Synthetic Data Generator API (scaffold)",
        "version": "0.1.0",
        "endpoints": {
            "gc_garch": {
                "generate": "/api/gc/generate",
                "status": "/api/status/{job_id}",
                "download": "/api/download/{job_id}",
            },
            "utility": {"health": "/health"},
        },
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}

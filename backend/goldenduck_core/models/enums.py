from enum import Enum


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobType(str, Enum):
    garch = "garch"
    gan = "gan"
    ddpm = "ddpm"

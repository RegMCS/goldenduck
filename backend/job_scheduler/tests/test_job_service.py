from unittest.mock import Mock

from job_scheduler.models.enums import JobStatus, JobType
from job_scheduler.services.job_service import create_job, update_job_status


def test_create_job_persists_and_returns_job():
    db = Mock()

    job = create_job(db=db, user_id="user-1", job_type=JobType.garch)

    assert job.requestor == "user-1"
    assert job.job_type == JobType.garch
    assert job.status == JobStatus.queued
    db.add.assert_called_once_with(job)
    db.commit.assert_called_once()
    db.refresh.assert_called_once_with(job)


def test_update_job_status_completed_sets_s3_and_timestamp():
    db = Mock()
    query = Mock()
    filtered = Mock()
    db.query.return_value = query
    query.filter_by.return_value = filtered

    update_job_status(
        db=db,
        job_id="job-1",
        status=JobStatus.completed,
        s3_url="s3://bucket/file.csv",
    )

    db.query.assert_called_once()
    query.filter_by.assert_called_once_with(id="job-1")
    values = filtered.update.call_args[0][0]
    assert values["status"] == JobStatus.completed
    assert values["s3_url"] == "s3://bucket/file.csv"
    assert "completed_at" in values
    db.commit.assert_called_once()


def test_update_job_status_non_completed_only_updates_status():
    db = Mock()
    query = Mock()
    filtered = Mock()
    db.query.return_value = query
    query.filter_by.return_value = filtered

    update_job_status(db=db, job_id="job-2", status=JobStatus.running)

    values = filtered.update.call_args[0][0]
    assert values == {"status": JobStatus.running}
    db.commit.assert_called_once()

from sqlalchemy import Column, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
import uuid

from job_scheduler.db.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(Text, unique=True, index=True)
    hashed_password = Column(Text)
    first_name = Column(Text)
    last_name = Column(Text)
    is_admin = Column(Boolean, nullable=False, default=False)

from sqlalchemy import Column, Text
from sqlalchemy.dialects.postgresql import UUID
import uuid

from backend_app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    first_name = Column(Text)
    last_name = Column(Text)

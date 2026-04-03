from pydantic import BaseModel
from typing import Optional
from uuid import UUID


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None


class UserCreate(BaseModel):
    username: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserResponse(BaseModel):
    id: UUID
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_admin: bool = False

    class Config:
        from_attributes = True


class UserAdminUpdate(BaseModel):
    is_admin: bool


class UserListResponse(BaseModel):
    users: list[UserResponse]


class TokenWithUser(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

"""
SQLAlchemy declarative base for all ORM models.
"""

from sqlalchemy.orm import DeclarativeBase, MappedAsDataclass


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass

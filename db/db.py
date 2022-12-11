import os

from sqlalchemy import Column, String, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

db = os.getenv("POSTGRES_DB")
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

# создаем коннект к базе
engine = create_engine(f'postgresql://{user}:{password}@localhost:5432/{db}')
connection = engine.connect()
Base = declarative_base()

class Weights(Base):
    __tablename__ = 'weights'

    model = Column(String, primary_key=True)
    parameters = Column(JSONB)

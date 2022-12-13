import os

from sqlalchemy import Column, String, create_engine
from sqlalchemy.dialects.postgresql import JSONB,BYTEA
from sqlalchemy.orm import declarative_base

db = "ml_db"#os.getenv("POSTGRES_DB")
user ="eingur"# os.getenv("POSTGRES_USER")
password = "pass"#os.getenv("POSTGRES_PASSWORD")

# создаем коннект к базе
engine = create_engine(f'postgresql://{user}:{password}@postgres:5432/{db}')#localhost
connection = engine.connect()
Base = declarative_base()

class Weights(Base):
    __tablename__ = 'weights'

    model = Column(String, primary_key=True)
    parameters = Column(BYTEA)

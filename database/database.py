from sqlalchemy import create_engine

DATABASE_URI = "sqlite+pysqlite:///:memory:"

engine = create_engine(DATABASE_URI, echo=True, future=True)

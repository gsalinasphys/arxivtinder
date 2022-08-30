from sqlalchemy import create_engine

DATABASE_URI = "sqlite+pysqlite:///static/sqlite.db"

engine = create_engine(DATABASE_URI, echo=True, future=True)

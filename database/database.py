from sqlalchemy import create_engine

DATABASE_URI = "sqlite+pysqlite:///local_files/sqlite.db"

engine = create_engine(DATABASE_URI, future=True)

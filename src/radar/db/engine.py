from logging import warning
import json

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from radar import ROOT

DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / 'radar.db'

engine = create_engine(f'sqlite:///{DB_PATH.as_posix()}', echo=True, json_serializer= lambda obj: json.dumps(obj, sort_keys=True, ensure_ascii=False))

@event.listens_for(engine, 'connect')
def enable_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute('PRAGMA foreign_keys=ON')
    if cursor.execute('PRAGMA foreign_keys').fetchone()[0] != 1:
        warning("Something went wrong whilst configuring: 'PRAGMA foreign_keys=ON'")
    cursor.close()

Session = sessionmaker(engine)
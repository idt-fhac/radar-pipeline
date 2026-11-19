from logging import warning
import json
import os

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from radar import ROOT

DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
# Test runs while building the pipeline go to the test database; production must be chosen
# explicitly ($env:RADAR_DB = "prod"), so nothing lands there by accident.
DATABASES = {'test': 'radar-test.db', 'prod': 'radar.db'}
DB_NAME = os.getenv('RADAR_DB', 'test')
if DB_NAME not in DATABASES:
    raise ValueError(f'RADAR_DB={DB_NAME!r} is unknown, use one of: {", ".join(DATABASES)}')
DB_PATH = DATA_DIR / DATABASES[DB_NAME]

# SQL logging: on by default while developing; the CLI switches it off unless --sql is given.
SQL_ECHO = os.getenv('RADAR_SQL_ECHO', '1') == '1'

engine = create_engine(f'sqlite:///{DB_PATH.as_posix()}', echo=SQL_ECHO, json_serializer= lambda obj: json.dumps(obj, sort_keys=True, ensure_ascii=False))

@event.listens_for(engine, 'connect')
def enable_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute('PRAGMA foreign_keys=ON')
    if cursor.execute('PRAGMA foreign_keys').fetchone()[0] != 1:
        warning("Something went wrong whilst configuring: 'PRAGMA foreign_keys=ON'")
    cursor.close()

Session = sessionmaker(engine)
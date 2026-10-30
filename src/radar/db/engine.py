from pathlib import Path
import json

Path('data').mkdir(exist_ok=True)

from sqlalchemy import create_engine

engine = create_engine('sqlite:///data/radar.db', echo=True, json_serializer= lambda obj: json.dumps(obj, sort_keys=True, ensure_ascii=False))

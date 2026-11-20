from sqlalchemy import MetaData, inspect
from sqlalchemy.orm import DeclarativeBase

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

MAX_REPR_LENGTH = 40

class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    def __repr__(self):
        """e.g. Post(id=3, container_id=1, external_id='Ug1...', text='Die Reichweite ist…')"""
        state = inspect(self)
        loaded = state.dict
        parts = []
        for attr in state.mapper.column_attrs:
            if attr.key not in loaded:
                value = '…'  # not loaded: reading it would trigger a query
            else:
                value = repr(loaded[attr.key])
                if len(value) > MAX_REPR_LENGTH:
                    value = value[:MAX_REPR_LENGTH] + '…'
            parts.append(f'{attr.key}={value}')
        return f'{type(self).__name__}({", ".join(parts)})'
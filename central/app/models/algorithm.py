# Deprecated: Algorithm model removed; algorithms are hardcoded services now.
from sqlalchemy import Column, Integer, String, LargeBinary

from ..db.database import Base

class Algorithm(Base):
    __tablename__ = "algorithms"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String)
    version = Column(String)
    schema = Column(String)
    central_file = Column(LargeBinary)
    remote_file = Column(LargeBinary)

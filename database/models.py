from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, create_engine, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from config.settings import Config

Base = declarative_base()

class Vehicle(Base):
    __tablename__ = 'vehicles'
    
    id = Column(Integer, primary_key=True)
    track_id = Column(Integer)
    license_plate = Column(String(20))
    confidence = Column(Float)
    first_seen = Column(DateTime, default=func.now())
    last_seen = Column(DateTime, default=func.now(), onupdate=func.now())
    processed = Column(Boolean, default=False)
    total_detections = Column(Integer, default=1)
    best_frame_path = Column(String(255), nullable=True)
    axle_count = Column(Integer, default=2)

    def to_dict(self):
        return {
            'id': self.id,
            'track_id': self.track_id,
            'license_plate': self.license_plate,
            'confidence': float(self.confidence),
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'processed': bool(self.processed),
            'total_detections': self.total_detections,
            'best_frame_path': self.best_frame_path,
            'axle_count': self.axle_count
        }

# Create tables
def init_db():
    engine = create_engine(Config.DATABASE_URL)
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    init_db()
# database/db_operations.py
from sqlalchemy import create_engine, event, Index, text
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import time
import logging
from datetime import datetime, timedelta
from .models import Base, Vehicle
from config.settings import Config
from sqlalchemy.sql import func

class DatabaseManager:
    def __init__(self):
        self.setup_logging()
        self.initialize_engine()
        self.setup_session()
        self.setup_engine_events()
        self.create_indices()

    def setup_logging(self):
        self.logger = logging.getLogger('database')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('logs/database.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    def initialize_engine(self):
        try:
            self.engine = create_engine(
                Config.DATABASE_URL,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30,
                pool_recycle=1800
            )
            Base.metadata.create_all(self.engine)
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise

    def create_indices(self):
        try:
            with self.engine.connect() as conn:
                # Create indices using text() to create proper SQL statements
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_license_plate 
                    ON vehicles (license_plate)
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_last_seen 
                    ON vehicles (last_seen)
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_track 
                    ON vehicles (track_id)
                """))
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_confidence 
                    ON vehicles (confidence)
                """))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to create indices: {str(e)}")
            raise

    def setup_session(self):
        self.Session = scoped_session(sessionmaker(bind=self.engine))

    def setup_engine_events(self):
        @event.listens_for(self.engine, 'connect')
        def receive_connect(dbapi_connection, connection_record):
            self.logger.info("Database connection established")

    @contextmanager
    def session_scope(self):
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()

    def add_vehicle_detection(self, track_id: int, license_plate: str,
                            confidence: float, frame_path: Optional[str] = None,
                            axle_count: int = 2) -> bool:
        try:
            with self.session_scope() as session:
                existing = session.query(Vehicle)\
                    .filter(Vehicle.track_id == track_id)\
                    .first()
                
                if existing:
                    existing.total_detections += 1
                    existing.last_seen = datetime.utcnow()
                    if confidence > existing.confidence:
                        existing.confidence = confidence
                        existing.license_plate = license_plate
                        existing.best_frame_path = frame_path
                        existing.axle_count = axle_count
                else:
                    vehicle = Vehicle(
                        track_id=track_id,
                        license_plate=license_plate,
                        confidence=confidence,
                        best_frame_path=frame_path,
                        axle_count=axle_count
                    )
                    session.add(vehicle)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding vehicle detection: {str(e)}")
            return False

    def get_recent_vehicles(self, minutes: int = 30) -> List[Dict]:
        try:
            with self.session_scope() as session:
                cutoff = datetime.utcnow() - timedelta(minutes=minutes)
                vehicles = session.query(Vehicle)\
                    .filter(Vehicle.last_seen >= cutoff)\
                    .order_by(Vehicle.last_seen.desc())\
                    .all()
                
                return [{
                    'id': v.id,
                    'license_plate': v.license_plate,
                    'confidence': float(v.confidence),
                    'first_seen': v.first_seen.isoformat(),
                    'last_seen': v.last_seen.isoformat(),
                    'processed': bool(v.processed),
                    'frame_path': v.best_frame_path,
                    'track_id': v.track_id,
                    'total_detections': v.total_detections,
                    'axle_count': v.axle_count
                } for v in vehicles]
                
        except Exception as e:
            self.logger.error(f"Error fetching recent vehicles: {str(e)}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        try:
            with self.session_scope() as session:
                total = session.query(Vehicle).count()
                processed = session.query(Vehicle)\
                    .filter_by(processed=True).count()
                
                avg_confidence = session.query(func.avg(Vehicle.confidence))\
                    .scalar() or 0
                
                recent = session.query(Vehicle)\
                    .filter(Vehicle.first_seen >= datetime.utcnow() - timedelta(hours=24))\
                    .count()
                
                return {
                    'total_vehicles': total,
                    'processed_vehicles': processed,
                    'average_confidence': float(avg_confidence),
                    'vehicles_last_24h': recent
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching statistics: {str(e)}")
            return {
                'total_vehicles': 0,
                'processed_vehicles': 0,
                'average_confidence': 0,
                'vehicles_last_24h': 0
            }

    def search_vehicles(self, plate_query: str) -> List[Dict]:
        try:
            with self.session_scope() as session:
                vehicles = session.query(Vehicle)\
                    .filter(Vehicle.license_plate.ilike(f'%{plate_query}%'))\
                    .order_by(Vehicle.last_seen.desc())\
                    .all()
                
                return [{
                    'id': v.id,
                    'license_plate': v.license_plate,
                    'confidence': float(v.confidence),
                    'first_seen': v.first_seen.isoformat(),
                    'last_seen': v.last_seen.isoformat(),
                    'processed': bool(v.processed),
                    'frame_path': v.best_frame_path,
                    'track_id': v.track_id,
                    'total_detections': v.total_detections,
                    'axle_count': v.axle_count
                } for v in vehicles]
                
        except Exception as e:
            self.logger.error(f"Error searching vehicles: {str(e)}")
            return []
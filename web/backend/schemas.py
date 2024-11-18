# web/backend/schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class VehicleBase(BaseModel):
    license_plate: str
    confidence: float
    track_id: int

class VehicleCreate(VehicleBase):
    pass

class VehicleResponse(VehicleBase):
    id: int
    first_seen: datetime
    last_seen: datetime
    processed: bool
    total_detections: int
    best_frame_path: Optional[str] = None

    class Config:
        from_attributes = True

class Statistics(BaseModel):
    total_vehicles: int
    processed_vehicles: int
    average_confidence: float
    vehicles_last_24h: int
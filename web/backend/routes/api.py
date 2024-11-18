from fastapi import APIRouter, HTTPException
from typing import List
from ..schemas import VehicleResponse, Statistics
from database.db_operations import DatabaseManager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

router = APIRouter()
db_manager = DatabaseManager()

@router.get("/vehicles/recent", response_model=List[VehicleResponse])
async def get_recent_vehicles():
    try:
        vehicles = db_manager.get_recent_vehicles(minutes=30)
        # Sort by confidence, highest first
        vehicles.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return vehicles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/statistics")
async def get_statistics():
    try:
        return db_manager.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/vehicles/search/{plate}")
async def search_vehicles(plate: str):
    try:
        return db_manager.search_vehicles(plate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/images/{filename}")
async def get_image(filename: str):
    try:
        plate_path = f"plates/{filename}"
        if os.path.exists(plate_path):
            return FileResponse(plate_path)
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
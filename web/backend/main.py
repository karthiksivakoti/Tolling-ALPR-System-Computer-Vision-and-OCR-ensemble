from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, Response
from pathlib import Path
import cv2
import uvicorn
import sys
import os
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional
from database.db_operations import DatabaseManager
from config.settings import Config
from core.detector import VehicleDetector
from core.tracker import VehicleTracker
from core.ocr_engine import OCREngine
from utils.roi_utils import ROIManager
import numpy as np
from .websocket import manager, ConnectionManager
from .middleware import RateLimiter
from utils.image_manager import image_manager

class VideoCamera:
    def __init__(self, source=0):
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
            
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.detector = VehicleDetector()
        self.tracker = VehicleTracker()
        self.ocr_engine = OCREngine()
        self.roi_manager = ROIManager()
        self.db_manager = DatabaseManager()
        
        self.frame_count = 0
        self.skip_frames = 2  
        
    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    async def get_frame(self) -> Optional[bytes]:
        if not self.cap.isOpened():
            return None
            
        self.frame_count += 1
        
        if self.frame_count % self.skip_frames != 0:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return self._encode_frame(frame)
            
        success, frame = self.cap.read()
        if not success:
            return None

        try:
            detections = self.detector.detect_and_track(frame)
            
            for detection in detections:
                if detection['class'] == 'license_plate':
                    bbox = detection['bbox']
                    detection_confidence = detection['confidence']
                    axle_count = detection.get('axle_count', 2)
                    
                    track = self.tracker.get_track(
                        detection['track_id'], 
                        bbox,
                        detection_confidence,
                        axle_count
                    )
                    
                    intersection = self.roi_manager.calculate_intersection(bbox)
                    in_roi = intersection > Config.ROI_INTERSECTION_THRESHOLD
                    
                    if track.should_process(in_roi):
                        plate_img = self.detector.extract_plate(frame, bbox)
                        
                        if plate_img is not None:
                            text, confidence = self.ocr_engine.process_plate(plate_img, track.track_id)
                            
                            if text and confidence > 0:
                                if track.update_plate(text, confidence, plate_img):
                                    frame_path = await image_manager.save_image(
                                        track.best_plate_image,
                                        text,
                                        track.track_id
                                    )
                                    
                                    if frame_path:
                                        track.frame_path = frame_path
                                        vehicle_data = await self.db_manager.add_vehicle_detection(
                                            track_id=track.track_id,
                                            license_plate=text,
                                            confidence=confidence,
                                            frame_path=frame_path,
                                            axle_count=axle_count
                                        )
                                        if vehicle_data:
                                            await manager.broadcast_vehicle_update(vehicle_data)
            
            if self.roi_manager.roi:
                frame = self.roi_manager.draw_roi(frame)
            frame = self.tracker.draw_tracks(frame)
            
            return self._encode_frame(frame)
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            return self._encode_frame(frame)

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        try:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return buffer.tobytes() if ret else None
        except Exception as e:
            print(f"Frame encoding error: {str(e)}")
            return None

app = FastAPI(title="License Plate Detection System")

app.middleware("http")(RateLimiter())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Content-Length"]
)

db_manager = DatabaseManager()
camera: Optional[VideoCamera] = None

app.mount("/plates", StaticFiles(directory="plates"), name="plates")

@app.on_event("startup")
async def startup_event():
    global camera
    video_source = 0
    if len(sys.argv) > 2 and sys.argv[1] == '-v':
        video_source = sys.argv[2]
    camera = VideoCamera(video_source)
    print(f"Application started, camera initialized with source: {video_source}")

@app.on_event("shutdown")
async def shutdown_event():
    global camera
    if camera:
        del camera
        camera = None
    print("Application shutting down, camera released")

async def gen_frames() -> AsyncIterator[bytes]:
    global camera
    while True:
        frame = await camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "License Plate Detection System API",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/vehicles/recent")
async def get_recent_vehicles():
    try:
        vehicles = await db_manager.get_recent_vehicles(minutes=30)
        await manager.broadcast_vehicle_update({"vehicles": vehicles})
        return vehicles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics")
async def get_statistics():
    try:
        stats = await db_manager.get_statistics()
        await manager.broadcast_statistics(stats)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vehicles/search/{plate}")
async def search_vehicles(plate: str):
    try:
        vehicles = await db_manager.search_vehicles(plate)
        return vehicles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/images/{filename}")
async def get_image(filename: str):
    try:
        image_data = await image_manager.get_image(filename)
        if image_data is None:
            raise HTTPException(status_code=404, detail="Image not found")
        return Response(content=image_data, media_type="image/jpeg")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/roi")
async def get_roi():
    try:
        if not camera or not camera.roi_manager:
            raise HTTPException(status_code=500, detail="Camera or ROI manager not initialized")
        return {"roi": camera.roi_manager.roi}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/roi")
async def set_roi(roi: dict):
    try:
        if not camera or not camera.roi_manager:
            raise HTTPException(status_code=500, detail="Camera or ROI manager not initialized")
        camera.roi_manager.roi = tuple(roi["points"])
        camera.roi_manager.save_roi()
        return {"status": "success", "roi": camera.roi_manager.roi}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/plates/{filename}")
async def delete_plate_image(filename: str):
    try:
        success = await image_manager.delete_image(filename)
        if not success:
            raise HTTPException(status_code=404, detail="Image not found")
        return {"status": "success", "message": f"Image {filename} deleted"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        log_level="info"
    )
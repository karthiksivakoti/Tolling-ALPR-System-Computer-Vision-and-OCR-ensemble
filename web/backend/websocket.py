from fastapi import WebSocket
from typing import List, Dict
import json
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_vehicle_data: Dict = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        if self.last_vehicle_data:
            await websocket.send_json(self.last_vehicle_data)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_vehicle_update(self, vehicle_data: dict):
        """Broadcast vehicle detection updates to all clients"""
        self.last_vehicle_data = vehicle_data
        for connection in self.active_connections.copy():
            try:
                await connection.send_json({
                    "type": "vehicle_update",
                    "data": vehicle_data
                })
            except Exception:
                await self.disconnect(connection)

    async def broadcast_statistics(self, stats_data: dict):
        """Broadcast statistics updates to all clients"""
        for connection in self.active_connections.copy():
            try:
                await connection.send_json({
                    "type": "statistics_update",
                    "data": stats_data
                })
            except Exception:
                await self.disconnect(connection)

# Create global instance
manager = ConnectionManager()
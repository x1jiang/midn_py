from fastapi import WebSocket
from typing import List, Dict

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, site_id: str):
        await websocket.accept()
        self.active_connections[site_id] = websocket

    def disconnect(self, websocket: WebSocket, site_id: str):
        if site_id in self.active_connections:
            del self.active_connections[site_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

    async def send_to_site(self, message: str, site_id: str):
        if site_id in self.active_connections:
            await self.active_connections[site_id].send_text(message)

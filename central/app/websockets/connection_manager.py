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
        print(f"🌐 ConnectionManager: Attempting to send message to site {site_id}")
        print(f"📡 ConnectionManager: Active connections: {list(self.active_connections.keys())}")
        
        if site_id in self.active_connections:
            try:
                websocket = self.active_connections[site_id]
                print(f"✅ ConnectionManager: Found WebSocket for site {site_id}")
                await websocket.send_text(message)
                print(f"📤 ConnectionManager: Successfully sent message to site {site_id}")
                print(f"📝 ConnectionManager: Message sent: {message[:100]}{'...' if len(message) > 100 else ''}")
            except Exception as e:
                print(f"💥 ConnectionManager: Error sending to site {site_id}: {e}")
                # Remove broken connection
                del self.active_connections[site_id]
        else:
            print(f"❌ ConnectionManager: Site {site_id} not in active connections")

"""Application event bus for WebSocket clients."""

import time
import uuid
from collections import deque
from typing import Any

from fastapi import WebSocket


class EventBus:
    """Keeps recent events and broadcasts them to connected clients."""

    def __init__(self, max_history: int = 200):
        self.clients: list[WebSocket] = []
        self.history = deque(maxlen=max_history)

    def register(self, ws: WebSocket) -> None:
        if ws not in self.clients:
            self.clients.append(ws)

    def unregister(self, ws: WebSocket) -> None:
        if ws in self.clients:
            self.clients.remove(ws)

    def get_history(self, count: int = 50) -> list[dict[str, Any]]:
        return list(self.history)[-count:]

    async def publish(
        self,
        event_type: str,
        data: Any = None,
        *,
        message: str | None = None,
        actor: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
            "message": message,
            "actor": actor,
        }
        self.history.append(event)

        dead_clients: list[WebSocket] = []
        for client in list(self.clients):
            try:
                await client.send_json(event)
            except Exception:
                dead_clients.append(client)

        for dead in dead_clients:
            self.unregister(dead)

        return event


event_bus = EventBus()

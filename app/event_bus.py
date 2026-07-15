"""Application event bus for WebSocket clients."""

import time
import uuid
from collections import deque
from typing import Any, Optional

from fastapi import WebSocket

from app.audit_log import append_audit, should_audit_event

# 一般ユーザー向け WebSocket から除外するイベント種別
ADMIN_ONLY_EVENT_TYPES = frozenset({"litellm_proxy_request"})


class EventBus:
    """Keeps recent events and broadcasts them to connected clients."""

    def __init__(self, max_history: int = 200):
        self.clients: list[tuple[WebSocket, dict]] = []
        self.history = deque(maxlen=max_history)

    def register(self, ws: WebSocket, user: dict) -> None:
        entry = (ws, user)
        if entry not in self.clients:
            self.clients.append(entry)

    def unregister(self, ws: WebSocket) -> None:
        self.clients = [(client, user) for client, user in self.clients if client is not ws]

    @staticmethod
    def sanitize_event_for_user(event: dict[str, Any], user: dict) -> Optional[dict[str, Any]]:
        if user.get("role") == "admin":
            return event
        event_type = str(event.get("type") or "")
        if event_type in ADMIN_ONLY_EVENT_TYPES:
            return None
        return event

    def get_history(self, count: int = 50, *, user: Optional[dict] = None) -> list[dict[str, Any]]:
        items = list(self.history)[-count:]
        if not user:
            return items
        sanitized: list[dict[str, Any]] = []
        for event in items:
            filtered = self.sanitize_event_for_user(event, user)
            if filtered is not None:
                sanitized.append(filtered)
        return sanitized

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

        if should_audit_event(event_type):
            append_audit(action=event_type, actor=actor, message=message, data=data)

        dead_clients: list[WebSocket] = []
        for client, user in list(self.clients):
            filtered = self.sanitize_event_for_user(event, user)
            if filtered is None:
                continue
            try:
                await client.send_json(filtered)
            except Exception:
                dead_clients.append(client)

        for dead in dead_clients:
            self.unregister(dead)

        return event


event_bus = EventBus()

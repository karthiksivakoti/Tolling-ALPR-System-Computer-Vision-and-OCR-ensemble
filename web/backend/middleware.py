# web/backend/middleware.py
import time
from fastapi import Request, Response
import redis
from typing import Callable
import json

class RateLimiter:
    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.rate_limit = 100  # requests per minute
        self.window = 60  # seconds

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host
        key = f"rate_limit:{client_ip}"
        
        # Skip rate limiting for WebSocket connections
        if "upgrade" in request.headers and request.headers["upgrade"].lower() == "websocket":
            return await call_next(request)
            
        current = self.redis.get(key)
        if current is None:
            self.redis.setex(key, self.window, 1)
        elif int(current) >= self.rate_limit:
            return Response(
                content=json.dumps({"detail": "Rate limit exceeded"}),
                status_code=429
            )
        else:
            self.redis.incr(key)
        
        return await call_next(request)
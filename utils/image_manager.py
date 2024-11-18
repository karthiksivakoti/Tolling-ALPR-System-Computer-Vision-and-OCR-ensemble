from pathlib import Path
import asyncio
import aiofiles
import aiofiles.os as async_os
from datetime import datetime, timedelta
import redis
from typing import Optional
import json
import cv2

class ImageManager:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=1)
        self.cache_ttl = 3600  # 1 hour
        self.cleanup_interval = 3600  # 1 hour
        self.max_age_days = 30
        self.image_dir = Path("plates")
        self.image_dir.mkdir(exist_ok=True)
        
        # Start cleanup task
        asyncio.create_task(self.cleanup_loop())

    async def save_image(self, frame, plate_text: str, track_id: int) -> Optional[str]:
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{plate_text}_{track_id}_{timestamp}.jpg"
            filepath = self.image_dir / filename

            # Save to disk
            cv2.imwrite(str(filepath), frame)
            
            # Cache in Redis
            _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_bytes = img_encoded.tobytes()
            
            self.redis_client.setex(
                f"image:{filename}",
                self.cache_ttl,
                image_bytes
            )
            
            return filename
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return None

    async def get_image(self, filename: str) -> Optional[bytes]:
        try:
            # Try cache first
            cached = self.redis_client.get(f"image:{filename}")
            if cached:
                return cached
            
            # Read from disk
            filepath = self.image_dir / filename
            if await async_os.path.exists(filepath):
                async with aiofiles.open(filepath, 'rb') as f:
                    data = await f.read()
                    
                # Update cache
                self.redis_client.setex(
                    f"image:{filename}",
                    self.cache_ttl,
                    data
                )
                return data
                
            return None
        except Exception as e:
            print(f"Error retrieving image: {str(e)}")
            return None

    async def cleanup_loop(self):
        while True:
            try:
                await self.cleanup_old_images()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                print(f"Cleanup error: {str(e)}")

    async def cleanup_old_images(self):
        cutoff = datetime.now() - timedelta(days=self.max_age_days)
        try:
            # Changed from async for to regular for
            for filepath in Path(self.image_dir).glob("*.jpg"):
                try:
                    stat = await async_os.stat(filepath)
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    
                    if mtime < cutoff:
                        await async_os.remove(filepath)
                        self.redis_client.delete(f"image:{filepath.name}")
                        print(f"Cleaned up old image: {filepath.name}")
                        
                except Exception as e:
                    print(f"Error cleaning up {filepath}: {str(e)}")
        except Exception as e:
            print(f"Error in cleanup: {str(e)}")

image_manager = ImageManager()
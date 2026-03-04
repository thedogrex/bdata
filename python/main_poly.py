from fastapi import FastAPI
from app.core import Core
import app.config
import asyncio
import os
import logging

from predictor.poly_client import PolymarketClient

os.environ['TZ'] = 'UTC'


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#host = '0.0.0.0' if not config.IS_LOCAL else '127.0.0.1'
port = 8503  # websocket port
host = '127.0.0.1'

app = FastAPI()

core = Core()

logging.getLogger('websockets.protocol').setLevel(logging.WARNING)

import time

timestamp_seconds = int(time.time())
print(timestamp_seconds)

@app.on_event('startup')
def app_startup():
    asyncio.ensure_future(core.run())

@app.get("/")
async def index():
    return {"message": "Initialized message from FastAPI! Version: v.0.11.2"}


if __name__ == '__main__':

    client = PolymarketClient()

    market = client.fetch_current_active_market()

    print("Slug:", market.slug)
    print("Timestamp:", market.timestamp)
    print("Ends:", market.end_date)
    print("Question:", market.question)
    print("Description:", market.description)
    print("Is Closed:", market.closed)

    for outcome in market.outcomes:
        print("Outcome:", outcome.name, "Asset:", outcome.asset_id)

    import uvicorn
    uvicorn.run(app, host=host, port=port)



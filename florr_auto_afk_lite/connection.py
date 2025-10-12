import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import json

app = FastAPI()

block_alpha = False
connected_clients = set()

@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    """处理 WebSocket 连接和消息"""
    global block_alpha
    
    try:

        await websocket.accept()
        
        connected_clients.add(websocket)
        print(f"Client connected, total clients: {len(connected_clients)}")
        
        response = {'action': 'setState', 'state': block_alpha}
        await websocket.send_text(json.dumps(response))
        
        try:
            while True:
                message = await websocket.receive_text()
                data = json.loads(message)
                action = data.get('action')
                
                if action == 'getState':

                    response = {'action': 'setState', 'state': block_alpha}
                    await websocket.send_text(json.dumps(response))
                    
                elif action == 'setState':

                    new_state = data.get('state')
                    if new_state is not None:
                        block_alpha = new_state
                        response = {'action': 'setState', 'state': block_alpha}
                        await broadcast_message(response)
        
        except WebSocketDisconnect:
            print(f"Client disconnected, remaining clients: {len(connected_clients) - 1}")
        finally:
            if websocket in connected_clients:
                connected_clients.remove(websocket)
    except Exception as e:
        print(f"Connection error: {type(e).__name__}: {e}")
        if websocket in connected_clients:
            connected_clients.remove(websocket)

async def broadcast_message(message):
    if connected_clients:
        data = json.dumps(message)
        tasks = [client.send_text(data) for client in connected_clients]
        await asyncio.gather(*tasks, return_exceptions=True)

async def get_state():
    return block_alpha

async def set_state(new_state):
    global block_alpha
    block_alpha = new_state
    
    response = {'action': 'setState', 'state': block_alpha}
    await broadcast_message(response)
    return True

async def main():
    config = uvicorn.Config(
        "connection:app",
        host="localhost",
        port=8765,
        log_level="info",
        reload=False
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
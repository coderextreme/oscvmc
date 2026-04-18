from pythonosc import dispatcher
from pythonosc import osc_server
import socketio
import json

# connect to JSONvirse
server_url = 'http://localhost:8088'
print(f"Connecting to {server_url}...")
sio = socketio.Client()

# Event handler for successful connection
@sio.event
def connect():
    print('Connected to server')
    print(f'Session ID: {sio.sid}')

# Event handler for disconnection
@sio.event
def disconnect():
    print('Disconnected from server')

# Event handler for connection errors
@sio.event
def connect_error(data):
    print(f'Connection error: {data}')

# Custom event handler - receives messages from server
@sio.on('message')
def on_message(data):
    print(f'Received message: {data}')

# Custom event handler - receives custom events
@sio.on('custom_event')
def on_custom_event(data):
    print(f'Received custom event: {data}')

sio.connect(server_url)
# hook up to VideoFeed session and room, and make it active
sio.emit('clientjoin', "VideoFeed")
sio.emit('clientactivesession', "yottzumm")
sio.emit('clientactivename', "VideoFeed")

def print_handler(unused_addr, *args):
    print(f'Received: {args}')
    sio.emit('python_clientavatar', args)

disp = dispatcher.Dispatcher()
disp.map('/VMC/Ext/Bone/Pos', print_handler)

server = osc_server.BlockingOSCUDPServer(('127.0.0.1', 39539), disp)
print('OSC Server listening on port 39539...')
server.serve_forever()

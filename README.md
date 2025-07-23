# Critical Region Object Throw Detection System

## Overview
This system detects fast-moving thrown objects (such as sacks, stones, or tools) entering a user-defined critical region in real-time video feeds. It is designed to minimize false positives from birds, animals, and natural environmental motion.

## Features
- Real-time video input from webcam or file
- Interactive ROI (Region of Interest) selection
- Background subtraction (MOG2)
- High-speed motion and size filtering
- Morphological noise reduction
- Visual and console alerts with event logging
- Snapshot saving on detection
- Robust error handling

## File Structure
- `main.py`: Main application and GUI
- `motion_detector.py`: Motion detection logic
- `roi_manager.py`: ROI selection and management
- `alert_system.py`: Alert and logging
- `config.py`: Configurable parameters
- `requirements.txt`: Dependencies

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the application:
   ```bash
   python main.py
   ```

## Usage
- Draw ROI with mouse on the video window
- 'R': Reset ROI
- 'Q': Quit
- 'S': Save current frame

## Notes
- Ensure a working webcam or video file is available
- Snapshots and ROI coordinates are saved in the working directory 

## Production Deployment
For running on a server, use Gunicorn with Eventlet for Flask-SocketIO apps. Install dependencies including gunicorn.

### Setup
1. Install Nginx for reverse proxy.
2. Run web interface:
   ```
   gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:5000 web_interface:app
   ```
3. Run dashboard:
   ```
   gunicorn --worker-class eventlet -w 1 -b 0.0.0.0:4000 dashboard:app
   ```

### Nginx Config Example
```
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }

    location /dashboard {
        proxy_pass http://127.0.0.1:4000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
    }
}
```

Disable debug mode in code for production. For scaling, use load balancer with sticky sessions.
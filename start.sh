#!/bin/bash

# Start the server in the background
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 &

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..30}; do
    if curl -s http://localhost:7860/ > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 1
done

# Run inference
echo "Starting inference..."
python inference.py

# Keep server running (wait for background process)
wait
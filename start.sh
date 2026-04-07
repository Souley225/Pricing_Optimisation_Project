#!/bin/bash
# Start both FastAPI backend and Streamlit frontend

# Start the API server in the background
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
echo "Waiting for API to start..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is ready!"
        break
    fi
    sleep 1
done

# Start Streamlit on port 7860 (HF Spaces default)
exec streamlit run src/ui/app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false

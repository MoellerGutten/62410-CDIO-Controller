import cv2
import os
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata

API_KEY = os.getenv("ROBOFLOW_API_KEY", "YOUR_NEW_API_KEY_HERE")

# Initialize client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

source = WebcamSource(resolution=(1280, 720))

# Configure streaming options
config = StreamConfig(
    stream_output=["visualization"], 
    data_output=["predictions"],      
    processing_timeout=3600,          
    requested_plan="webrtc-gpu-medium", 
    requested_region="us"             
)

# Create streaming session
# Note: Ensure "workflow" name is the actual ID from your Roboflow dashboard
session = client.webrtc.stream(
    source=source,
    workflow="find-balls-and-balls",
    workspace="augusts-workspace",
    image_input="image",
    config=config
)

# Handle incoming video frames
@session.on_frame
def show_frame(frame, metadata):
    cv2.imshow("Roboflow WebRTC Output", frame)
    # If 'q' is pressed, we stop the session
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.stop() # Using .stop() for a cleaner exit

# Handle prediction data
@session.on_data
def on_data(data: dict, metadata: VideoMetadata):
    # Print the prediction count or labels to keep console clean
    predictions = data.get("predictions", [])
    print(f"Frame {metadata.frame_id} | Objects Detected: {len(predictions)}")

# Start the session
try:
    session.start() # Use .start() to initiate the WebRTC handshake
except Exception as e:
    print(f"Session failed to start: {e}")
finally:
    cv2.destroyAllWindows()
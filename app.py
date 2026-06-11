import os
import time
import io
import base64
import asyncio
from collections import deque
import multiprocessing
from typing import Optional, Dict, List
import warnings
import subprocess 
import uuid

warnings.filterwarnings('ignore', category=RuntimeWarning)

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
import uvicorn

# CPU optimization
num_cores = multiprocessing.cpu_count()
#cv2.setNumThreads(num_cores)
cv2.setNumThreads(1)
os.environ.setdefault("OMP_NUM_THREADS", str(num_cores))
os.environ.setdefault("MKL_NUM_THREADS", str(num_cores))
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|framedrop;1|buffer_size;512000|stimeout;5000000")

app = FastAPI(title="PPE Detection Backend", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
model_path = None
video_capture = None
processing_active = False

# Class colors (BGR)
CLASS_COLORS = {
    "helmet": (0, 200, 0),
    "vest": (0, 200, 0),
    "no-helmet": (0, 0, 255),
    "no-vest": (0, 140, 255),
    "person": (60, 220, 220),
}
DEFAULT_COL = (200, 200, 200)

def convert_video_for_pi(input_path: str):
    """
    Convert uploaded video into Raspberry Pi friendly format
    Uses lower resolution, frame rate, and codec settings optimized for Pi
    """

    output_path = f"/tmp/{uuid.uuid4().hex}_pi.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,

        # Resize to lower resolution + reduce FPS for Pi performance
        "-vf", "scale=640:360,fps=10",

        # Pi friendly H.264 encoding with baseline profile
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "3.0",
        
        # Additional flags for better compatibility
        "-movflags", "+faststart",
        "-max_muxing_queue_size", "1024",

        # Lower quality = smaller file size and faster processing
        "-crf", "28",

        # Remove audio to reduce file size
        "-an",

        output_path
    ]

    print(f"Converting video for Raspberry Pi: {input_path}")
    print(f"Output will be: {output_path}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        print("Video conversion completed successfully")
        return output_path
    except subprocess.TimeoutExpired:
        raise Exception("Video conversion timed out (>5 minutes)")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr}")
        raise Exception(f"Video conversion failed: {e.stderr}")
    except FileNotFoundError:
        raise Exception("FFmpeg not found. Please install ffmpeg on your Raspberry Pi: sudo apt-get install ffmpeg")

class ProcessingConfig(BaseModel):
    """Configuration for video processing"""
    # Model settings
    model_path: str = "models/ppe.onnx"
    model_mode: str = "PPE"  # "PPE" or "Default"
    conf_threshold: float = 0.5
    imgsz: int = 384
    
    # Video source
    video_source: Optional[str] = None
    video_source_type: str = "RTSP"  # "RTSP" or "File"
    backend: str = "FFmpeg"  # "FFmpeg" or "GStreamer"
    rotate_180: bool = False
    
    # Zones (only for Default mode)
    roi_tl_x: int = 465
    roi_tl_y: int = 334
    roi_br_x: int = 1390
    roi_br_y: int = 1075
    ng_tl_x: int = 670
    ng_tl_y: int = 14
    ng_br_x: int = 1385
    ng_br_y: int = 310
    
    # Privacy settings
    use_pixelation: bool = True
    gaussian_kernel: int = 31
    gaussian_sigma: int = 15
    blur_person_only: bool = True
    
    # Display
    show_text_labels: bool = True


class FrameResponse(BaseModel):
    """Response for processed frame"""
    frame_base64: str
    detections: List[Dict]
    entries_roi: int
    entries_nogo: int
    fps: float
    person_in_nogo: bool
    inference_time_ms: float


def load_model_if_needed(weights_path: str):
    """Load model if not already loaded or if path changed"""
    global model, model_path
    
    # Convert to absolute path first for consistent comparison
    if not os.path.isabs(weights_path):
        weights_path = os.path.join(os.path.dirname(__file__), weights_path)
    
    # Only load if model is None or path changed
    if model is None or model_path != weights_path:
        print(f"Loading model: {weights_path}")
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model file not found: {weights_path}")
        
        model = YOLO(weights_path, task='detect')  # Explicitly set task to avoid warnings
        model_path = weights_path
        print(f"Model loaded successfully: {weights_path}")
    
    return model


def color_for(class_name: str):
    return CLASS_COLORS.get(class_name, DEFAULT_COL)


def in_rect(cx, cy, tl, br):
    return tl[0] <= cx <= br[0] and tl[1] <= cy <= br[1]


def open_video_capture(url: str, mode: str):
    """Open video capture with appropriate backend"""
    if mode.startswith("GStreamer"):
        gst = (
            f"rtspsrc location={url} protocols=tcp latency=200 drop-on-latency=true ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
        )
        return cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


def read_video_with_ffmpeg(video_path: str):
    """
    Generator that yields frames from video using ffmpeg directly
    This bypasses OpenCV's video codec issues on Raspberry Pi
    """
    import subprocess
    
    # Get video info first
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets",
        "-of", "json",
        video_path
    ]
    
    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        import json as json_lib
        probe_data = json_lib.loads(probe_result.stdout)
        stream = probe_data['streams'][0]
        width = int(stream['width'])
        height = int(stream['height'])
        
        print(f"Video info: {width}x{height}")
    except Exception as e:
        print(f"Could not probe video: {e}")
        width, height = 640, 360  # Default
    
    # FFmpeg command to extract frames as raw RGB
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-"
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8
    )
    
    frame_size = width * height * 3  # 3 bytes per pixel (BGR)
    
    while True:
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            break
        
        # Convert raw bytes to numpy array
        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        
        yield frame
    
    process.stdout.close()
    process.wait()


def process_frame(frame: np.ndarray, config: ProcessingConfig, 
                prev_in_roi: Dict, prev_in_nogo: Dict,
                entries_roi: int, entries_nogo: int):
    """Process a single frame with detection, zones, and blur"""
    
    # Load model
    current_model = load_model_if_needed(config.model_path)
    
    # Rotate if needed
    if config.rotate_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    PPE_MODE = (config.model_mode == "PPE")
    
    # Determine classes to detect
    det_classes = None if PPE_MODE else [0]  # person class
    
    # Run inference
    t_inf_start = time.perf_counter()
    results = current_model.predict(
        frame,
        conf=config.conf_threshold,
        verbose=False,
        imgsz=config.imgsz,
        classes=det_classes,
    )
    t_inf_end = time.perf_counter()
    inference_time_ms = (t_inf_end - t_inf_start) * 1000
    
    res = results[0]
    annotated = frame.copy()
    
    # Draw zones (only in Default mode)
    if not PPE_MODE:
        ROI_TL = (config.roi_tl_x, config.roi_tl_y)
        ROI_BR = (config.roi_br_x, config.roi_br_y)
        NG_TL = (config.ng_tl_x, config.ng_tl_y)
        NG_BR = (config.ng_br_x, config.ng_br_y)
        
        cv2.rectangle(annotated, ROI_TL, ROI_BR, (0, 255, 255), 2)
        overlay = annotated.copy()
        cv2.rectangle(overlay, NG_TL, NG_BR, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)
        cv2.rectangle(annotated, NG_TL, NG_BR, (0, 0, 255), 2)
    
    # Process detections
    person_in_nogo = False
    detections = []
    boxes = res.boxes
    
    if boxes is not None:
        ids = list(range(len(boxes)))
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        clss = boxes.cls.int().cpu().tolist() if boxes.cls is not None else []
        
        for tid, (x1, y1, x2, y2), conf, cls_id in zip(ids, xyxy, confs, clss):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = current_model.names.get(cls_id, str(cls_id))
            col = color_for(class_name)
            
            detections.append({
                "id": tid,
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy]
            })
            
            if PPE_MODE:
                # PPE MODE: show all detections
                cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 2)
                if config.show_text_labels:
                    cv2.putText(annotated, f"{class_name} {conf:.2f} ID {tid}",
                            (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, col, 2, cv2.LINE_AA)
                continue
            
            # DEFAULT MODE (zones/counters/blur)
            in_ng = in_rect(cx, cy, NG_TL, NG_BR)
            in_roi = in_rect(cx, cy, ROI_TL, ROI_BR)
            
            if not (in_ng or in_roi):
                prev_in_nogo[tid] = False
                prev_in_roi[tid] = False
                continue
            
            # Count entries for 'person'
            if class_name == "person":
                was_in_ng = prev_in_nogo.get(tid, False)
                was_in_roi = prev_in_roi.get(tid, False)
                if (not was_in_ng) and in_ng:
                    entries_nogo += 1
                if (not was_in_roi) and in_roi:
                    entries_roi += 1
                prev_in_nogo[tid] = in_ng
                prev_in_roi[tid] = in_roi
            
            # Privacy blur
            if config.blur_person_only and class_name != "person":
                pass
            else:
                if in_ng or in_roi:
                    crop = annotated[y1:y2, x1:x2]
                    if crop.size > 0:
                        if config.use_pixelation:
                            ph = max(1, (y2 - y1) // 20)
                            pw = max(1, (x2 - x1) // 20)
                            small = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_LINEAR)
                            pixel = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                            annotated[y1:y2, x1:x2] = pixel
                        else:
                            kernel = (config.gaussian_kernel, config.gaussian_kernel)
                            annotated[y1:y2, x1:x2] = cv2.GaussianBlur(crop, kernel, config.gaussian_sigma)
            
            # Draw boxes
            if in_ng:
                person_in_nogo = person_in_nogo or (class_name == "person")
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            elif in_roi:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), col, 1)
            
            # Labels
            if config.show_text_labels:
                label = f"{class_name} {conf:.2f} ID {tid}"
                cv2.putText(annotated, label, (x1, max(25, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,0,255) if in_ng else col, 2, cv2.LINE_AA)
    
    # Add alerts and counters (only in Default mode)
    if not PPE_MODE:
        if person_in_nogo:
            text = "ALERT: NO-GO VIOLATION!"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
            x = max(10, (annotated.shape[1] - tw) // 2)
            y = 50
            cv2.rectangle(annotated, (x-10, y-th-10), (x+tw+10, y+10), (0,0,255), -1)
            cv2.putText(annotated, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255,255,255), 3, cv2.LINE_AA)
        
        cv2.putText(annotated, f"Entries (ROI) [person]: {entries_roi}",
                (ROI_TL[0], max(20, ROI_TL[1]-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"NO-GO Entries [person]: {entries_nogo}",
                (NG_TL[0], max(30, NG_TL[1]-12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    
    return annotated, detections, entries_roi, entries_nogo, person_in_nogo, inference_time_ms


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "PPE Detection Backend v2.0",
        "model_loaded": model is not None,
        "current_model": model_path,
        "processing_active": processing_active
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": model_path,
        "cpu_cores": num_cores,
        "processing_active": processing_active
    }


@app.websocket("/ws/process")
async def websocket_process(websocket: WebSocket):
    """WebSocket endpoint for real-time video processing"""
    await websocket.accept()
    
    global processing_active
    processing_active = True
    
    # State for tracking
    prev_in_roi = {}
    prev_in_nogo = {}
    entries_roi = 0
    entries_nogo = 0
    
    t_last = time.time()
    frames_count = 0
    fps = 0.0
    
    try:
        # Receive configuration
        config_data = await websocket.receive_json()
        config = ProcessingConfig(**config_data)
        
        # Open video capture
        if config.video_source is None:
            await websocket.send_json({"error": "No video source provided"})
            return
            
        use_ffmpeg_direct = False
        frame_generator = None
        
        if config.video_source_type == "File":
            print(f"Original video file: {config.video_source}")
            
            # Check if file exists
            if not os.path.exists(config.video_source):
                await websocket.send_json({"error": f"Video file not found: {config.video_source}"})
                return
            
            print(f"Opening video file: {config.video_source}")
            
            # Try multiple backends for better compatibility on Raspberry Pi
            cap = None
            backends_to_try = [
                ("FFMPEG", cv2.CAP_FFMPEG),
                ("GSTREAMER", cv2.CAP_GSTREAMER),
                ("V4L2", cv2.CAP_V4L2),
                ("ANY", cv2.CAP_ANY)
            ]
            
            for backend_name, backend_flag in backends_to_try:
                print(f"Trying {backend_name} backend...")
                try:
                    cap = cv2.VideoCapture(config.video_source, backend_flag)
                    if cap.isOpened():
                        # Try to read a test frame
                        test_ok, test_frame = cap.read()
                        if test_ok and test_frame is not None:
                            print(f"Successfully opened with {backend_name} backend")
                            print(f"Video resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
                            # Reset to beginning
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            break
                        else:
                            print(f"{backend_name} opened but cannot read frames")
                            cap.release()
                            cap = None
                    else:
                        print(f"{backend_name} backend failed to open")
                        cap = None
                except Exception as e:
                    print(f"{backend_name} backend error: {e}")
                    if cap:
                        cap.release()
                    cap = None
            
            if cap is None or not cap.isOpened():
                print("⚠️ All OpenCV backends failed. Trying direct ffmpeg frame extraction...")
                try:
                    # Test if ffmpeg can read the video
                    frame_generator = read_video_with_ffmpeg(config.video_source)
                    test_frame = next(frame_generator)
                    if test_frame is not None and test_frame.size > 0:
                        print(f"✅ FFmpeg direct extraction working! Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
                        use_ffmpeg_direct = True
                        # Recreate generator for actual processing
                        frame_generator = read_video_with_ffmpeg(config.video_source)
                    else:
                        raise Exception("FFmpeg could not extract frames")
                except Exception as e:
                    print(f"❌ FFmpeg direct extraction also failed: {e}")
                    await websocket.send_json({
                        "error": "Failed to open video with OpenCV or FFmpeg. Please check: 1) FFmpeg is installed, 2) Video codec is supported, 3) File is not corrupted."
                    })
                    return
            else:
                # Set optimal properties for Raspberry Pi
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            print(f"Opening video source: {config.video_source} with backend {config.backend}")
            cap = open_video_capture(config.video_source, config.backend)
            if not cap.isOpened():
                await websocket.send_json({"error": f"Failed to open video source: {config.video_source}"})
                return
        
        print(f"Starting video processing loop...")
        frame_num = 0
        
        while processing_active:
            # Read frame based on method
            if use_ffmpeg_direct:
                try:
                    frame = next(frame_generator)
                    ok = True
                except StopIteration:
                    ok = False
                    frame = None
                    print("FFmpeg generator exhausted - end of video")
            else:
                ok, frame = cap.read()
            
            if not ok or frame is None:
                if not use_ffmpeg_direct:
                    print(f"Failed to read frame. ok={ok}, frame is None={frame is None}")
                
                if config.video_source_type == "File":
                    if use_ffmpeg_direct:
                        print("Reached end of video file (ffmpeg)")
                        await websocket.send_json({"status": "complete"})
                        break
                    else:
                        # Check if we've reached end of file
                        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        print(f"Video position: {current_pos}/{total_frames}")
                        
                        if current_pos >= total_frames - 1:
                            print("Reached end of video file")
                            await websocket.send_json({"status": "complete"})
                            break
                        else:
                            print("Failed to read frame but not at end of file - video may be corrupted or codec issue")
                            await websocket.send_json({"error": "Unable to read video frames. The video codec may not be supported on this device."})
                            break
                else:
                    # Try to reconnect for RTSP
                    print("Attempting to reconnect to RTSP stream...")
                    cap.release()
                    await asyncio.sleep(0.5)
                    cap = open_video_capture(config.video_source, config.backend)
                    continue
            
            frame_num += 1
            if frame_num % 30 == 0:
                print(f"Processing frame {frame_num}...")
            
            # Process frame
            annotated, detections, entries_roi, entries_nogo, person_in_nogo, inf_time = process_frame(
                frame, config, prev_in_roi, prev_in_nogo, entries_roi, entries_nogo
            )
            
            # Calculate FPS
            frames_count += 1
            now = time.time()
            if now - t_last >= 1.0:
                fps = frames_count / (now - t_last)
                frames_count = 0
                t_last = now
            
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            # Send response
            response = {
                "frame": frame_base64,
                "detections": detections,
                "entries_roi": entries_roi,
                "entries_nogo": entries_nogo,
                "fps": round(fps, 1),
                "person_in_nogo": person_in_nogo,
                "inference_time_ms": round(inf_time, 1)
            }
            
            if frame_num % 30 == 0:
                print(f"Sending frame {frame_num} to frontend...")
            
            await websocket.send_json(response)
            
            # Small delay to prevent overwhelming the connection
            await asyncio.sleep(0.01)
        
        cap.release()
        
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error in websocket: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        processing_active = False


@app.post("/stop")
async def stop_processing():
    """Stop video processing"""
    global processing_active
    processing_active = False
    return {"status": "stopped"}


if __name__ == "__main__":
    # Backend runs on port 8000 (default)
    # Change port here if needed: uvicorn.run(app, host="0.0.0.0", port=5000)
    uvicorn.run(app, host="0.0.0.0", port=8009)

# Made with Bob

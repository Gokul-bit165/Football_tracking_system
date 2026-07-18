import cv2
import numpy as np
from ultralytics import YOLO
from src.team_classification.color_classifier import ColorTeamClassifier
from src.tracking.kalman_ball_filter import KalmanBallFilter

def run_visualization_pipeline(video_path: str, output_path: str, model_path: str):
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    # Configure custom HSV reference colors for this specific game
    # Team A: White kits (Low Saturation, High Value)
    # Team B: Red kits (Hue near 0 or 180, High Saturation)
    # Referee: Yellow kits (Hue near 30)
    custom_colors = {
        "Team A": np.array([0, 15, 240], dtype=np.float32),      # White
        "Team B": np.array([0, 240, 200], dtype=np.float32),     # Red
        "Referee": np.array([30, 200, 180], dtype=np.float32)    # Yellow
    }
    
    color_classifier = ColorTeamClassifier(team_colors=custom_colors)
    kf_filter = KalmanBallFilter(dt=0.04, process_noise=1.5, measurement_noise=1.5)
    
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Keep track of the ball trajectory tail
    ball_history = []
    max_tail_len = 30
    consecutive_missed_ball = 0
    
    print(f"Processing video: {video_path} ({width}x{height} @ {fps:.2f} FPS) -> {output_path}")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Run YOLO tracking with ByteTrack
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        
        # If no detections, write original frame
        if len(results) == 0 or results[0].boxes is None:
            out.write(frame)
            frame_idx += 1
            continue
            
        boxes = results[0].boxes
        
        # Extract predictions
        xyxys = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        
        # Get tracking IDs if available, else use placeholder
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [0] * len(classes)
            
        # Collect ball detection candidates
        ball_candidates = []
        
        # 2. Draw players, goalkeeper, and referee
        for i in range(len(classes)):
            cls_id = classes[i]
            x1, y1, x2, y2 = map(int, xyxys[i])
            track_id = track_ids[i]
            conf = confidences[i]
            
            # Draw player class
            if cls_id == 2:  # player
                # Crop upper 45% (jersey region)
                crop = frame[y1:y2, x1:x2]
                
                # Restrict classification only to Team A (White) vs Team B (Red)
                dominant_hsv = color_classifier.get_dominant_hsv(crop)
                dist_A = color_classifier.hsv_distance(dominant_hsv, custom_colors["Team A"])
                dist_B = color_classifier.hsv_distance(dominant_hsv, custom_colors["Team B"])
                
                if dist_A < dist_B:
                    box_color = (255, 255, 255)  # White
                    label = f"White {track_id}"
                else:
                    box_color = (0, 0, 255)        # Red
                    label = f"Red {track_id}"
                    
                # Draw box and label (decreased font size)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Label box tag
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                y_label_start = max(y1, label_size[1] + 10)
                cv2.rectangle(frame, (x1, y_label_start - label_size[1] - 5), (x1 + label_size[0], y_label_start + 2), box_color, cv2.FILLED)
                text_color = (0, 0, 0) if box_color == (255, 255, 255) else (255, 255, 255)
                cv2.putText(frame, label, (x1, y_label_start - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
                
            elif cls_id == 1:  # goalkeeper
                box_color = (255, 255, 0)  # Cyan
                label = f"GK {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                y_label_start = max(y1, label_size[1] + 10)
                cv2.rectangle(frame, (x1, y_label_start - label_size[1] - 5), (x1 + label_size[0], y_label_start + 2), box_color, cv2.FILLED)
                cv2.putText(frame, label, (x1, y_label_start - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                
            elif cls_id == 3:  # referee
                box_color = (0, 255, 0)  # Green
                label = f"Ref {track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                y_label_start = max(y1, label_size[1] + 10)
                cv2.rectangle(frame, (x1, y_label_start - label_size[1] - 5), (x1 + label_size[0], y_label_start + 2), box_color, cv2.FILLED)
                cv2.putText(frame, label, (x1, y_label_start - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
            elif cls_id == 0:  # ball
                ball_candidates.append((conf, (x1, y1, x2, y2)))
                
        # 3. Kalman Filter update and draw for ball position (only draw if actively detected)
        if len(ball_candidates) > 0:
            # Select candidate with highest confidence
            ball_candidates.sort(reverse=True, key=lambda x: x[0])
            best_box = ball_candidates[0][1]
            bx = (best_box[0] + best_box[2]) / 2.0
            by = (best_box[1] + best_box[3]) / 2.0
            
            # Smooth with Kalman Filter
            smoothed_pt = kf_filter.update([bx, by])
            cx, cy = int(smoothed_pt[0]), int(smoothed_pt[1])
            
            # Draw glowing dot on the current ball position
            cv2.circle(frame, (cx, cy), 5, (0, 140, 255), -1)
            cv2.circle(frame, (cx, cy), 8, (0, 180, 255), 1)
            
        # Write output frame
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed frame {frame_idx}/{total_frames}...")
            
    cap.release()
    out.release()
    print("Inference completed successfully!")

if __name__ == "__main__":
    run_visualization_pipeline(
        video_path="Copy of A1606b0e6_0 (14).mp4",
        output_path="output_visualization.mp4",
        model_path="runs/detect/runs/detect/yolo11m_baseline-4/weights/best.pt"
    )

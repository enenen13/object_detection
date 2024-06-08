from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # n for nano, you can use other sizes like yolov8s.pt, yolov8m.pt, etc.

# Global variable to store the click position
click_position = None

def gen_frames():
    global click_position
    cap = cv2.VideoCapture(0)  # Capture video from the first camera device
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Perform object detection
            results = model(frame)
            result_img = results[0].plot()  # Render the results

            if click_position:
                x, y = click_position
                for result in results:
                    for box in result.boxes:
                        if box.xyxy[0] <= x <= box.xyxy[2] and box.xyxy[1] <= y <= box.xyxy[3]:
                            label = result.names[int(box.cls)]
                            cv2.putText(result_img, f'{label}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', result_img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Concatenate frame data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/click_position', methods=['POST'])
def get_click_position():
    global click_position
    data = request.json
    click_position = (data['x'], data['y'])
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)



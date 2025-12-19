import cv2
from flask import Flask, render_template, Response, request
from fer import FER
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

app = Flask(__name__)

# Khởi tạo mô hình nhận diện
detector = FER(mtcnn=True) 

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Hàm vẽ chữ Tiếng Việt TO và RÕ
def draw_vn_text(img, text, position):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # Bạn nên upload file arial.ttf lên cùng thư mục để chữ to đẹp hơn
        font = ImageFont.truetype("arial.ttf", 45) 
    except:
        font = ImageFont.load_default()
        
    draw.text(position, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

emotion_dict = {
    "happy": "Hạnh phúc 😊",
    "sad": "Buồn 😢",
    "angry": "Giận dữ 😡",
    "surprise": "Ngạc nhiên 😲",
    "fear": "Sợ hãi 😨",
    "disgust": "Ghê tởm 🤢",
    "neutral": "Bình thường 😐"
}

# SỬA LẠI HÀM NÀY ĐỂ TẮT/BẬT CAMERA
def generate_frames():
    # Camera chỉ được mở khi người dùng nhấn nút "Bật" trên web
    camera = cv2.VideoCapture(0)
    
    # Kiểm tra nếu không mở được camera
    if not camera.isOpened():
        print("Không thể kết nối camera")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                results = detector.detect_emotions(frame)
                for result in results:
                    (x, y, w, h) = result["box"]
                    emotion_type = max(result["emotions"], key=result["emotions"].get)
                    label_vn = emotion_dict.get(emotion_type, "Đang quét...")
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    # Chữ hiển thị to và rõ phía trên khung
                    frame = draw_vn_text(frame, label_vn, (x, y - 60))

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b' \r\n')
    finally:
        # KHI TẮT TRÊN WEB, HÀM NÀY SẼ DỪNG VÀ GIẢI PHÓNG CAMERA NGAY LẬP TỨC
        camera.release()
        print("Camera đã được tắt và giải phóng.")

@app.route('/', methods=['GET', 'POST'])
def index():
    label = None
    image_path = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            img = cv2.imread(path)
            results = detector.detect_emotions(img)
            if results:
                emotion_type = max(results[0]["emotions"], key=results[0]["emotions"].get)
                label = emotion_dict.get(emotion_type, "Không rõ")
                (x, y, w, h) = results[0]["box"]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)
                img = draw_vn_text(img, label, (x, y - 70))
                cv2.imwrite(path, img)
                image_path = path
    return render_template('index.html', label=label, image_path=image_path)

@app.route('/video_feed')
def video_feed():
    # Luồng stream chỉ bắt đầu khi route này được gọi
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
from flask import Flask, render_template, request, send_from_directory, jsonify
import gc
import time
import os
import cv2
import numpy as np
from moviepy import ImageSequenceClip
from werkzeug.utils import secure_filename
import threading
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
UPLOAD_FOLDER = 'Uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Gerenciamento de progresso por tarefa
progress_dict = {}
progress_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=4)

def update_progress(task_id, value):
    with progress_lock:
        progress_dict[task_id] = value

def get_progress_value(task_id):
    with progress_lock:
        return progress_dict.get(task_id, 0)

def pixelate_frame(frame, pixel_size=64, color_clusters=-1):
    small = cv2.resize(frame, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    if color_clusters == -1:
        return cv2.resize(small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    Z = small.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, color_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(small.shape)
    return cv2.resize(quantized, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

def process_video(task_id, input_path, output_path, color_clusters):
    cap = cv2.VideoCapture(input_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(pixelate_frame(frame, color_clusters=color_clusters))
            update_progress(task_id, ((i + 1) / frame_count) * 80)
    finally:
        cap.release()
        gc.collect()
        time.sleep(1.5)

    def simulate_write_progress():
        while get_progress_value(task_id) < 99:
            with progress_lock:
                current = progress_dict[task_id]
                progress_dict[task_id] = min(current + 1, 99)
            time.sleep(0.5)

    sim_thread = threading.Thread(target=simulate_write_progress)
    sim_thread.start()

    clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=fps)
    clip.write_videofile(output_path, codec="libx264")

    with progress_lock:
        progress_dict[task_id] = 100

    sim_thread.join()

    try:
        os.remove(input_path)
    except PermissionError:
        print(f"Não foi possível deletar {input_path}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files['video']
        if video:
            # Verificar tamanho do arquivo
            max_size_mb = 100  # Limite de 100 MB
            if int(request.content_length or 0) > max_size_mb * 1024 * 1024:
                return jsonify(error="O arquivo é muito grande. O tamanho máximo permitido é 100 MB."), 400

            # Verificar extensão do arquivo
            allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            file_extension = os.path.splitext(video.filename)[1].lower()
            if file_extension not in allowed_extensions:
                return jsonify(error="Formato de arquivo inválido. Por favor, envie um vídeo nos formatos: mp4, avi, mov, mkv ou webm"), 400

            try:
                shutil.rmtree(PROCESSED_FOLDER)
            except Exception:
                pass
            os.makedirs(PROCESSED_FOLDER, exist_ok=True)

            # Usar sempre 32 bits (sem quantização de cores)
            color_clusters = -1

            unique_id = str(uuid.uuid4())
            filename = secure_filename(video.filename)
            filename = f'{unique_id}_{filename}'
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            output_filename = f'pixel_{filename}'
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)

            for file in os.listdir(UPLOAD_FOLDER):
                try:
                    os.remove(os.path.join(UPLOAD_FOLDER, file))
                except PermissionError:
                    pass

            video.save(upload_path)
            update_progress(unique_id, 0)

            executor.submit(process_video, unique_id, upload_path, output_path, color_clusters)
            return jsonify(download_link=output_filename, task_id=unique_id)

    return render_template('index.html')

@app.route('/processed/<filename>')
def download_file(filename):
    try:
        return send_from_directory(PROCESSED_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify(error="Arquivo não encontrado"), 404

@app.route('/progress/<task_id>')
def get_progress(task_id):
    return jsonify(progress=get_progress_value(task_id))

if __name__ == '__main__':
    app.run(debug=True)
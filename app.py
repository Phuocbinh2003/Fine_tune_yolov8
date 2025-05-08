import os
import cv2
import streamlit as st
from ultralytics import YOLO
import datetime
import shutil
import uuid
from glob import glob

# ================= CONFIG =================
UPLOAD_DIR = "temp_data"
IMG_DIR = os.path.join(UPLOAD_DIR, "images")
LABEL_DIR = os.path.join(UPLOAD_DIR, "labels")
PERMANENT_MODEL_DIR = "models"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(PERMANENT_MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="YOLOv8 Fine-tune", layout="centered")
st.title("🧠 YOLOv8 Fine-tuning Tool")

# ================= SESSION STATE =================
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "image_files" not in st.session_state:
    st.session_state.image_files = []

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Cài đặt")

# Model và ảnh
duploaded_model = st.sidebar.file_uploader("Tải lên model (.pt)", type=["pt"])
uploaded_images = st.sidebar.file_uploader("Tải lên ảnh", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Thông số fine-tune
nc = st.sidebar.number_input("Số lớp", min_value=1, value=1)
class_names = st.sidebar.text_input("Tên lớp (cách nhau bằng phẩy)", value="BIB")
epochs = st.sidebar.slider("Epochs", 1, 100, 5)
batch_size = st.sidebar.selectbox("Batch size", [8, 16, 32, 64], index=1)
img_size = st.sidebar.number_input("Kích thước ảnh", 320, 1280, 640, step=32)
conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.4)

# ================= XỬ LÝ UPLOAD =================
def save_uploaded_files():
    if uploaded_model:
        model_path = os.path.join(UPLOAD_DIR, uploaded_model.name)
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.session_state.model_path = model_path

    if uploaded_images:
        for img in uploaded_images:
            unique_name = f"{uuid.uuid4().hex}_{img.name}"
            img_path = os.path.join(IMG_DIR, unique_name)
            with open(img_path, "wb") as f:
                f.write(img.getbuffer())
            st.session_state.image_files.append(unique_name)

# ================= ANNOTATION =================
def annotation_interface():
    idx = st.session_state.current_index
    total = len(st.session_state.image_files)

    if idx < total:
        img_name = st.session_state.image_files[idx]
        img_path = os.path.join(IMG_DIR, img_name)

        st.write(f"**Ảnh {idx+1}/{total}:** {img_name}")
        try:
            results = st.session_state.model.predict(img_path, conf=conf_thres)
            annotated = results[0].plot()
            st.image(annotated, use_container_width=True)

            col1, col2, col3 = st.columns([1,1,1])
            if col1.button("👍 Chấp nhận", key=f"accept_{idx}"):
                save_labels(results, img_name)
                next_image()
            if col2.button("👎 Bỏ qua", key=f"skip_{idx}"):
                next_image()
            if idx > 0 and col3.button("◀️ Quay lại", key=f"back_{idx}"):
                st.session_state.current_index -= 1
                st.rerun()

        except Exception as e:
            st.error(f"Lỗi xử lý ảnh: {e}")
            next_image()
    else:
        st.success("✅ Hoàn thành gán nhãn!")
        create_yaml_file()
        show_train_button()


def save_labels(results, img_name):
    label_file = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    with open(label_file, "w") as f:
        for box in results[0].boxes:
            cls = int(box.cls.item())
            x, y, w, h = box.xywhn[0].tolist()
            f.write(f"{cls} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")


def next_image():
    st.session_state.current_index += 1
    st.rerun()

# ================= DATASET YAML =================
def create_yaml_file():
    yaml_path = os.path.join(UPLOAD_DIR, "dataset.yaml")
    names_list = [n.strip() for n in class_names.split(',')]
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(UPLOAD_DIR)}\n")
        f.write("train: images\nval: images\n")
        f.write(f"nc: {nc}\nnames: {names_list}\n")
    st.session_state.yaml_path = yaml_path

# ================= TRAINING =================
def show_train_button():
    if st.button("🚀 Bắt đầu huấn luyện"):
        with st.spinner("Đang huấn luyện..."):
            try:
                model = YOLO(st.session_state.model_path)
                model.train(
                    data=st.session_state.yaml_path,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    project="runs",
                    exist_ok=True
                )
                save_final_model()
                st.success("Huấn luyện thành công!")
                offer_model_download()
            except Exception as e:
                st.error(f"Lỗi huấn luyện: {e}")


def save_final_model():
    runs = sorted(glob("runs/exp*/weights/best.pt"), key=os.path.getmtime)
    if runs:
        source = runs[-1]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(PERMANENT_MODEL_DIR, f"model_{timestamp}.pt")
        shutil.copy(source, dest)
        st.session_state.final_model = dest


def offer_model_download():
    with open(st.session_state.final_model, "rb") as f:
        st.download_button(
            label="⬇️ Tải model đã huấn luyện",
            data=f,
            file_name=os.path.basename(st.session_state.final_model),
            mime="application/octet-stream"
        )

# ================= MAIN =================
if __name__ == "__main__":
    save_uploaded_files()
    if "model_path" in st.session_state:
        try:
            st.session_state.model = YOLO(st.session_state.model_path)
            if st.session_state.image_files:
                annotation_interface()
        except Exception as e:
            st.error(f"Không thể tải model: {e}")
    else:
        st.info("Vui lòng tải lên model và ảnh để bắt đầu")

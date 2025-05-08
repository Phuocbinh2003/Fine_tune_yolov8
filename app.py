import os
import uuid
import shutil
import datetime
import streamlit as st
from glob import glob
from ultralytics import YOLO

# ================= CONFIG =================
UPLOAD_DIR = "temp_data"
IMG_DIR = os.path.join(UPLOAD_DIR, "images")
LABEL_DIR = os.path.join(UPLOAD_DIR, "labels")
TRAIN_IMG_DIR = os.path.join(UPLOAD_DIR, "train_images")
TRAIN_LABEL_DIR = os.path.join(UPLOAD_DIR, "train_labels")
PERMANENT_MODEL_DIR = "models"

# Tạo thư mục
for d in [IMG_DIR, LABEL_DIR, TRAIN_IMG_DIR, TRAIN_LABEL_DIR, PERMANENT_MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

st.set_page_config(page_title="YOLOv8 Fine-tune", layout="centered")
st.title("🧫 YOLOv8 Fine-tuning Tool")

# ================= SESSION =================
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "image_files" not in st.session_state:
    st.session_state.image_files = []
if "accepted" not in st.session_state:
    st.session_state.accepted = []

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Cài đặt")
# Input
uploaded_model = st.sidebar.file_uploader("Tải lên model (.pt)", type=["pt"])
uploaded_images = st.sidebar.file_uploader("Tải lên ảnh", type=["jpg","png","jpeg"], accept_multiple_files=True)
# Hyperparams
nc = st.sidebar.number_input("Số lớp", 1, 100, 1)
class_names = st.sidebar.text_input("Tên lớp (phân tách bởi dấu phẩy)", "BIB")
epochs = st.sidebar.slider("Epochs", 1, 100, 5)
batch_size = st.sidebar.selectbox("Batch size", [8,16,32,64], index=1)
img_size = st.sidebar.number_input("Kích thước ảnh", 320, 1280, 640, step=32)
conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.4)

# ================= UPLOAD HANDLER =================
def save_uploads():
    if uploaded_model:
        path = os.path.join(UPLOAD_DIR, uploaded_model.name)
        with open(path, 'wb') as f:
            f.write(uploaded_model.getbuffer())
        st.session_state.model_path = path
    if uploaded_images:
        for img in uploaded_images:
            name = f"{uuid.uuid4().hex}_{img.name}"
            dest = os.path.join(IMG_DIR, name)
            with open(dest, 'wb') as f:
                f.write(img.getbuffer())
            st.session_state.image_files.append(name)

# ================= ANNOTATION UI =================
def annotate():
    idx = st.session_state.current_index
    files = st.session_state.image_files
    total = len(files)
    if idx < total:
        fname = files[idx]
        st.write(f"**Ảnh {idx+1}/{total}:** {fname}")
        img_path = os.path.join(IMG_DIR, fname)
        try:
            results = st.session_state.model.predict(img_path, conf=conf_thres)
            anno = results[0].plot()
            st.image(anno, use_container_width=True)
            c1, c2 = st.columns(2)
            if c1.button("👍 Chấp nhận", key=f"acc_{idx}"):
                save_label_and_copy(results, fname)
                st.session_state.accepted.append(fname)
                next_img()
            if c2.button("👎 Bỏ qua", key=f"sk_{idx}"):
                next_img()
        except Exception as e:
            st.error(f"Lỗi: {e}")
            next_img()
    else:
        if len(st.session_state.accepted) == 0:
            st.warning("Không có ảnh nào được chấp nhận để huấn luyện.")
        else:
            st.success(f"Hoàn thành! Chọn được {len(st.session_state.accepted)} ảnh để fine-tune.")
            prepare_dataset()
            train_button()


def save_label_and_copy(results, fname):
    # save predicted labels
    base = os.path.splitext(fname)[0]
    label_path = os.path.join(TRAIN_LABEL_DIR, base + ".txt")
    with open(label_path, 'w') as f:
        for box in results[0].boxes:
            c = int(box.cls)
            x,y,w,h = box.xywhn[0].tolist()
            f.write(f"{c} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")
    # copy image
    shutil.copy(os.path.join(IMG_DIR, fname), os.path.join(TRAIN_IMG_DIR, fname))


def next_img():
    st.session_state.current_index += 1
    st.rerun()

# ================= DATASET =================
def prepare_dataset():
    yaml = os.path.join(UPLOAD_DIR, 'dataset.yaml')
    names = [n.strip() for n in class_names.split(',')]
    with open(yaml, 'w') as f:
        f.write(f"path: {os.path.abspath(UPLOAD_DIR)}\n")
        f.write("train: train_images\n")
        f.write("val: train_images\n")
        f.write(f"nc: {nc}\nnames: {names}\n")
    st.session_state.yaml = yaml

# ================= TRAIN =================
def train_button():
    if st.button("🚀 Start Fine-tune", help="Huấn luyện trên ảnh đã chọn"):
        with st.spinner("Fine-tuning..."):
            try:
                model = YOLO(st.session_state.model_path)
                model.train(
                    data=st.session_state.yaml,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    project='runs', exist_ok=True
                )
                save_final()
                st.success("Fine-tune hoàn tất!")
                download()
            except Exception as e:
                st.error(f"Lỗi training: {e}")


def save_final():
    runs = sorted(glob('runs/exp*/weights/best.pt'), key=os.path.getmtime)
    if runs:
        src = runs[-1]
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dst = os.path.join(PERMANENT_MODEL_DIR, f'model_{ts}.pt')
        shutil.copy(src, dst)
        st.session_state.final = dst


def download():
    with open(st.session_state.final, 'rb') as f:
        st.download_button('⬇️ Tải model', data=f, file_name=os.path.basename(st.session_state.final))

# ================= MAIN =================
if __name__ == '__main__':
    save_uploads()
    if 'model_path' in st.session_state and st.session_state.image_files:
        try:
            st.session_state.model = YOLO(st.session_state.model_path)
            annotate()
        except Exception as e:
            st.error(f"Không thể load model: {e}")
    else:
        st.info("Vui lòng tải model và ảnh để bắt đầu.")

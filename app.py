import os
import random
import shutil
import streamlit as st
from ultralytics import YOLO
import datetime

# ================= CONFIG =================
UPLOAD_DIR = "temp_data"  # Thư mục tạm cho dữ liệu
IMG_DIR = os.path.join(UPLOAD_DIR, "images")
LABEL_DIR = os.path.join(UPLOAD_DIR, "labels")
TRAIN_DIR = os.path.join(UPLOAD_DIR, "train")
VAL_DIR = os.path.join(UPLOAD_DIR, "val")
PERMANENT_MODEL_DIR = "models"  # Thư mục model cố định

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(PERMANENT_MODEL_DIR, exist_ok=True)

# ================= CÀI ĐẶT =================
st.set_page_config(page_title="YOLOv8 Fine-tune", layout="centered")
st.title("🧠 YOLOv8 Fine-tuning Tool")

# ================= SIDEBAR =================
uploaded_model = st.sidebar.file_uploader("Tải lên model (.pt)", type=["pt"])
uploaded_images = st.sidebar.file_uploader("Tải lên ảnh", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
uploaded_labels = st.sidebar.file_uploader("Tải lên nhãn (.txt)", type=["txt"], accept_multiple_files=True)

# Hyperparameter
nc = st.sidebar.number_input("Số lớp", min_value=1, value=1)
class_names = st.sidebar.text_input("Tên lớp (cách nhau bằng phẩy)", value="BIB")
epochs = st.sidebar.slider("Epochs", 1, 100, 5)
batch_size = st.sidebar.selectbox("Batch size", [8, 16, 32, 64], index=1)
img_size = st.sidebar.number_input("Kích thước ảnh", 320, 1280, 640, step=32)

# ================= XỬ LÝ DỮ LIỆU =================
def save_uploaded_files():
    """Lưu các file đã tải lên vào thư mục tạm và chia tách thành train/val"""
    if uploaded_model:
        model_path = os.path.join(UPLOAD_DIR, uploaded_model.name)
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.session_state.model_path = model_path

    if uploaded_images and uploaded_labels:
        # Lưu ảnh và nhãn vào thư mục tạm
        for img, label in zip(uploaded_images, uploaded_labels):
            img_path = os.path.join(IMG_DIR, img.name)
            label_path = os.path.join(LABEL_DIR, label.name)
            
            # Lưu ảnh
            with open(img_path, "wb") as f:
                f.write(img.getbuffer())
                
            # Lưu nhãn
            with open(label_path, "wb") as f:
                f.write(label.getbuffer())
        
        # Chia tách ảnh và nhãn thành train và val (80% train, 20% val)
        image_files = [img.name for img in uploaded_images]
        label_files = [label.name for label in uploaded_labels]
        
        # Xáo trộn danh sách ảnh và nhãn
        combined = list(zip(image_files, label_files))
        random.shuffle(combined)
        train_data = combined[:int(0.8 * len(combined))]  # 80% cho train
        val_data = combined[int(0.8 * len(combined)):]  # 20% cho val

        # Di chuyển ảnh và nhãn vào thư mục train và val
        for img, label in train_data:
            shutil.move(os.path.join(IMG_DIR, img), os.path.join(TRAIN_DIR, img))
            shutil.move(os.path.join(LABEL_DIR, label), os.path.join(TRAIN_DIR, label))
            
        for img, label in val_data:
            shutil.move(os.path.join(IMG_DIR, img), os.path.join(VAL_DIR, img))
            shutil.move(os.path.join(LABEL_DIR, label), os.path.join(VAL_DIR, label))

        # Cập nhật session state
        st.session_state.image_files = image_files
        st.session_state.train_images = [img for img, _ in train_data]
        st.session_state.val_images = [img for img, _ in val_data]
        st.session_state.train_labels = [label for _, label in train_data]
        st.session_state.val_labels = [label for _, label in val_data]

# ================= TẠO FILE dataset.yaml =================
def create_yaml_file():
    """Tạo file cấu hình dataset"""
    # Lấy đường dẫn tuyệt đối đến các thư mục train và val
    abs_train_path = os.path.abspath(TRAIN_DIR)
    abs_val_path = os.path.abspath(VAL_DIR)

    # Tạo file dataset.yaml với đường dẫn tuyệt đối
    yaml_path = os.path.join(UPLOAD_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""train: {abs_train_path}
val: {abs_val_path}
nc: {nc}
names: {[name.strip() for name in class_names.split(',')]}
""")
    st.session_state.yaml_path = yaml_path

# ================= HUẤN LUYỆN =================
def train_model():
    """Huấn luyện mô hình YOLOv8"""
    with st.spinner("Đang huấn luyện..."):
        try:
            model = YOLO(st.session_state.model_path)
            model.train(
                data=st.session_state.yaml_path,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                project="runs",
                name="exp",
                exist_ok=True
            )
            save_final_model()
            st.success("Huấn luyện thành công!")
            offer_model_download()
        except Exception as e:
            st.error(f"Lỗi huấn luyện: {str(e)}")

def save_final_model():
    """Lưu model vào thư mục cố định"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    source_path = os.path.join("runs", "exp", "weights", "best.pt")
    dest_path = os.path.join(PERMANENT_MODEL_DIR, f"model_{timestamp}.pt")
    shutil.copy(source_path, dest_path)
    st.session_state.final_model = dest_path

def offer_model_download():
    """Hiển thị nút tải model"""
    with open(st.session_state.final_model, "rb") as f:
        st.download_button(
            label="⬇️ Tải model đã huấn luyện",
            data=f,
            file_name=os.path.basename(st.session_state.final_model),
            mime="application/octet-stream"
        )

# ================= MAIN FLOW =================
if __name__ == "__main__":
    save_uploaded_files()
    
    if "model_path" in st.session_state and "image_files" in st.session_state:
        create_yaml_file()  # Tạo file dataset.yaml
        train_model()  # Bắt đầu huấn luyện
    else:
        st.info("Vui lòng tải lên model, ảnh và nhãn để bắt đầu")

import os
import cv2
import streamlit as st
from ultralytics import YOLO
import datetime
import shutil

# ================= CONFIG =================
UPLOAD_DIR = "temp_data"  # Thư mục tạm cho dữ liệu
IMG_DIR = os.path.join(UPLOAD_DIR, "images")
LABEL_DIR = os.path.join(UPLOAD_DIR, "labels")
PERMANENT_MODEL_DIR = "models"  # Thư mục model cố định

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(PERMANENT_MODEL_DIR, exist_ok=True)

st.set_page_config(page_title="YOLOv8 Fine-tune", layout="centered")
st.title("🧠 YOLOv8 Fine-tuning Tool")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Cài đặt")
uploaded_model = st.sidebar.file_uploader("Tải lên model (.pt)", type=["pt"])
uploaded_images = st.sidebar.file_uploader("Tải lên ảnh", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Hyperparameter
nc = st.sidebar.number_input("Số lớp", min_value=1, value=1)
class_names = st.sidebar.text_input("Tên lớp (cách nhau bằng phẩy)", value="BIB")
epochs = st.sidebar.slider("Epochs", 1, 100, 5)
batch_size = st.sidebar.selectbox("Batch size", [8, 16, 32, 64], index=1)
img_size = st.sidebar.number_input("Kích thước ảnh", 320, 1280, 640, step=32)

# ================= XỬ LÝ DỮ LIỆU =================
def save_uploaded_files():
    """Lưu các file đã tải lên vào thư mục tạm"""
    if uploaded_model:
        model_path = os.path.join(UPLOAD_DIR, uploaded_model.name)
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.session_state.model_path = model_path

    if uploaded_images:
        for img in uploaded_images:
            img_path = os.path.join(IMG_DIR, img.name)
            with open(img_path, "wb") as f:
                f.write(img.getbuffer())
        st.session_state.image_files = [img.name for img in uploaded_images]

# ================= HIỂN THỊ ANNOTATION =================
def annotation_interface():
    """Giao diện chọn nhãn"""
    idx = st.session_state.get("current_index", 0)
    
    if idx < len(st.session_state.image_files):
        img_name = st.session_state.image_files[idx]
        img_path = os.path.join(IMG_DIR, img_name)
        
        try:
            # Dự đoán và hiển thị kết quả
            results = st.session_state.model.predict(img_path, conf=0.4)
            annotated_img = results[0].plot()  # Sử dụng built-in plotting
            st.image(annotated_img, caption=f"Ảnh {idx+1}/{len(st.session_state.image_files)}", use_container_width =True)
            
            # Nút điều khiển
            col1, col2 = st.columns(2)
            if col1.button("👍 Chấp nhận", key=f"accept_{idx}"):
                save_labels(results, img_name)
                next_image()
            
            if col2.button("👎 Bỏ qua", key=f"skip_{idx}"):
                next_image()
                
        except Exception as e:
            st.error(f"Lỗi xử lý ảnh: {str(e)}")
            next_image()
    else:
        st.success("✅ Hoàn thành gán nhãn!")
        create_yaml_file()
        show_train_button()

def save_labels(results, img_name):
    """Lưu nhãn vào file txt"""
    label_file = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + ".txt")
    with open(label_file, "w") as f:
        for box in results[0].boxes:
            cls = int(box.cls.item())
            xywhn = box.xywhn[0].tolist()
            f.write(f"{cls} {xywhn[0]:.5f} {xywhn[1]:.5f} {xywhn[2]:.5f} {xywhn[3]:.5f}\n")

def next_image():
    """Chuyển sang ảnh tiếp theo"""
    st.session_state.current_index += 1
    st.experimental_rerun()

# ================= HUẤN LUYỆN =================
def create_yaml_file():
    """Tạo file cấu hình dataset"""
    yaml_path = os.path.join(UPLOAD_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""train: {IMG_DIR}
val: {IMG_DIR}
nc: {nc}
names: {[name.strip() for name in class_names.split(',')]}
""")
    st.session_state.yaml_path = yaml_path

def show_train_button():
    """Hiển thị nút huấn luyện"""
    if st.button("🚀 Bắt đầu huấn luyện", help="Nhấn để bắt đầu quá trình training"):
        with st.spinner("Đang huấn luyện..."):
            try:
                model = YOLO(st.session_state.model_path)
                results = model.train(
                    data=st.session_state.yaml_path,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    project="runs",
                    name="exp",
                    exist_ok=True
                )
                
                # Di chuyển model đã huấn luyện
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
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    save_uploaded_files()
    
    if "model_path" in st.session_state:
        try:
            st.session_state.model = YOLO(st.session_state.model_path)
            if "image_files" in st.session_state:
                annotation_interface()
        except Exception as e:
            st.error(f"Không thể tải model: {str(e)}")
    else:
        st.info("Vui lòng tải lên model và ảnh để bắt đầu")
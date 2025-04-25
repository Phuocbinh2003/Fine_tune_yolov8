import os
import zipfile
import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image

st.sidebar.title("🔧 Fine-tune YOLOv8")

# Upload model
model_file = st.sidebar.file_uploader("Upload pretrained YOLOv8 model (.pt)", type=["pt"])
if model_file:
    os.makedirs("uploads", exist_ok=True)
    model_path = os.path.join("uploads", model_file.name)
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())
    st.sidebar.success("Model uploaded.")

# Upload image dataset (zip)
zip_file = st.sidebar.file_uploader("Upload image dataset (.zip)", type=["zip"])
img_dir = "uploads/images"
if zip_file:
    os.makedirs(img_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(img_dir)
    st.sidebar.success("Images extracted.")

# Params
epochs = st.sidebar.number_input("Epochs", 1, value=5)
batch_size = st.sidebar.number_input("Batch Size", 1, value=16)
img_size = st.sidebar.number_input("Image Size", 32, value=640)
label_dir = "uploads/labels"
project_dir = "fine_tune"
run_name = "run"

os.makedirs(label_dir, exist_ok=True)

# Load button
if st.sidebar.button("Load"):
    if not model_file or not zip_file:
        st.sidebar.error("Please upload model and dataset.")
    else:
        st.session_state['model'] = YOLO(model_path)
        st.session_state['images'] = [f for f in os.listdir(img_dir) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
        st.session_state['idx'] = 0
        st.success("Model & Images loaded.")

# Draw box
def draw_boxes(img_path, result):
    img = cv2.imread(img_path)
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Annotation flow
if 'images' in st.session_state:
    idx = st.session_state['idx']
    imgs = st.session_state['images']
    if idx < len(imgs):
        name = imgs[idx]
        path = os.path.join(img_dir, name)
        result = st.session_state['model'].predict(path, conf=0.3)[0]
        vis = draw_boxes(path, result)
        st.image(vis, caption=name, use_column_width=True)

        col1, col2 = st.columns(2)
        if col1.button("✅ Đúng", key=f"ok_{idx}"):
            with open(os.path.join(label_dir, name.rsplit('.',1)[0]+'.txt'), 'a') as f:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    xc, yc, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            st.session_state['idx'] += 1
            st.experimental_rerun()
        if col2.button("❌ Sai", key=f"skip_{idx}"):
            st.session_state['idx'] += 1
            st.experimental_rerun()
    else:
        st.success("🎉 Annotation hoàn tất.")
        # YAML
        yaml_path = os.path.join(project_dir, run_name, "data.yaml")
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            f.write(f"train: {img_dir}\nval: {img_dir}\nnc: 1\nnames: ['BIB']\n")
        st.write("✅ Created data.yaml")

        if st.button("🚀 Bắt đầu huấn luyện"):
            with st.spinner("Training..."):
                st.session_state['model'].train(
                    data=yaml_path,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    project=project_dir,
                    name=run_name,
                    exist_ok=True
                )
            st.success("✅ Training xong.")
            best_model = os.path.join(project_dir, run_name, "weights/best.pt")
            if os.path.isfile(best_model):
                with open(best_model, "rb") as f:
                    st.download_button("⬇️ Tải model đã huấn luyện", f, file_name="best.pt")

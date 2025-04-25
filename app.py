import os
import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image

# === CONFIG ===
UPLOAD_DIR = "./uploads"
IMG_DIR = os.path.join(UPLOAD_DIR, "images")
LABEL_DIR = os.path.join(UPLOAD_DIR, "labels")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

st.set_page_config(page_title="YOLOv8 Fine-tune", layout="centered")
st.title("🧠 Fine-tune YOLOv8 from Uploaded Images")

# === SIDEBAR ===
st.sidebar.header("Upload Data")

uploaded_model = st.sidebar.file_uploader("Upload YOLOv8 Model (.pt)", type=["pt"])
uploaded_images = st.sidebar.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

epochs = st.sidebar.number_input("Epochs", min_value=1, value=5)
batch_size = st.sidebar.number_input("Batch size", min_value=1, value=16)
img_size = st.sidebar.number_input("Image size", min_value=32, value=640)

project_dir = st.sidebar.text_input("Project output directory", "./runs")
run_name = st.sidebar.text_input("Run name", "exp")

# === SAVE MODEL + IMAGES ===
if uploaded_model:
    model_path = os.path.join(UPLOAD_DIR, uploaded_model.name)
    with open(model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    st.sidebar.success(f"Model saved to {model_path}")
    st.session_state["model_path"] = model_path

if uploaded_images:
    for img in uploaded_images:
        img_path = os.path.join(IMG_DIR, img.name)
        with open(img_path, "wb") as f:
            f.write(img.getbuffer())
    st.sidebar.success(f"Uploaded {len(uploaded_images)} images.")
    st.session_state["image_files"] = [f.name for f in uploaded_images]
    st.session_state["current_index"] = 0

# === LOAD MODEL ===
if "model_path" in st.session_state:
    try:
        model = YOLO(st.session_state["model_path"])
        st.session_state["model"] = model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

# === DRAW BOXES ===
def draw_boxes(img_path, results):
    img = cv2.imread(img_path)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === ANNOTATION ===
if "image_files" in st.session_state and "model" in st.session_state:
    idx = st.session_state.get("current_index", 0)
    image_files = st.session_state["image_files"]

    if idx < len(image_files):
        img_name = image_files[idx]
        img_path = os.path.join(IMG_DIR, img_name)

        # Predict
        result = st.session_state["model"].predict(img_path, conf=0.3)[0]
        annotated_img = draw_boxes(img_path, result)
        st.image(annotated_img, caption=img_name, use_column_width=True)
        st.write(f"Image {idx+1} / {len(image_files)}")

        col1, col2 = st.columns(2)
        if col1.button("✅ Đúng", key=f"yes_{idx}"):
            label_path = os.path.join(LABEL_DIR, img_name.rsplit(".", 1)[0] + ".txt")
            with open(label_path, "a") as f:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    xc, yc, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            st.session_state["current_index"] += 1
            st.rerun()

        if col2.button("❌ Sai", key=f"no_{idx}"):
            st.session_state["current_index"] += 1
            st.rerun()

    else:
        st.success("🎉 Annotation xong!")
        # Tạo data.yaml
        yaml_path = os.path.join(UPLOAD_DIR, "data.yaml")
        with open(yaml_path, "w") as f:
            f.write(f"""
train: {IMG_DIR}
val: {IMG_DIR}
nc: 1
names: ['BIB']
""")
        st.code(open(yaml_path).read(), language="yaml")
        if st.button("🚀 Bắt đầu huấn luyện"):
            with st.spinner("Training in progress..."):
                st.session_state["model"].train(
                    data=yaml_path,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    project=project_dir,
                    name=run_name,
                    exist_ok=True
                )
            st.success("✅ Training xong!")

            # Nút tải model về
            weight_path = os.path.join(project_dir, run_name, "weights", "best.pt")
            if os.path.exists(weight_path):
                with open(weight_path, "rb") as wf:
                    st.download_button(
                        label="⬇️ Tải model đã fine-tune",
                        data=wf,
                        file_name="fine_tuned_model.pt",
                        mime="application/octet-stream"
                    )
            else:
                st.warning("Không tìm thấy model đã huấn luyện.")
else:
    st.info("⬅️ Hãy upload model và ảnh để bắt đầu.")

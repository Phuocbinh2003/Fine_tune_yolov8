import os
import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image

# --- Sidebar Configuration ---
st.sidebar.title("Fine-tune YOLO on Streamlit with Upload/Download")

# Upload model
uploaded_model = st.sidebar.file_uploader("Upload pretrained model (.pt)", type=["pt"])
MODEL_PATH = None
if uploaded_model:
    MODEL_PATH = os.path.join("./uploads", uploaded_model.name)
    os.makedirs("./uploads", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        f.write(uploaded_model.getbuffer())
    st.sidebar.success(f"Model saved to {MODEL_PATH}")

# Upload images
uploaded_images = st.sidebar.file_uploader(
    "Upload images for annotation", type=["jpg", "png", "jpeg"], accept_multiple_files=True
)
IMG_DIR = "./uploads/images"
if uploaded_images:
    os.makedirs(IMG_DIR, exist_ok=True)
    for img in uploaded_images:
        img_path = os.path.join(IMG_DIR, img.name)
        with open(img_path, "wb") as f:
            f.write(img.getbuffer())
    st.sidebar.success(f"Saved {len(uploaded_images)} images.")

# Label output directory
temp_labels_dir = st.sidebar.text_input("Labels output directory", "./uploads/labels_temp")
os.makedirs(temp_labels_dir, exist_ok=True)

# Training parameters
epochs = st.sidebar.number_input("Epochs", min_value=1, value=5)
batch_size = st.sidebar.number_input("Batch size", min_value=1, value=16)
img_size = st.sidebar.number_input("Image size", min_value=32, value=640)
project_dir = st.sidebar.text_input("Project directory for output", "./fine_tune")
run_name = st.sidebar.text_input("Run name", "run")

# Load model & images button
if st.sidebar.button("Load Model & Images"):
    if not MODEL_PATH:
        st.sidebar.error("Please upload a .pt model first.")
    else:
        # Load model
        st.session_state['model'] = YOLO(MODEL_PATH)
        # List images
        imgs = []
        if os.path.isdir(IMG_DIR):
            imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        st.session_state['images'] = imgs
        st.session_state['idx'] = 0
        st.sidebar.success(f"Loaded model and {len(imgs)} images.")

# Helper: draw boxes
def draw_boxes_on_image(img_path, result):
    img = cv2.imread(img_path)
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Interactive annotation
if 'images' in st.session_state:
    idx = st.session_state['idx']
    images = st.session_state['images']
    if idx < len(images):
        img_name = images[idx]
        img_path = os.path.join(IMG_DIR, img_name)
        model = st.session_state['model']
        result = model.predict(img_path, conf=0.3)[0]
        annotated = draw_boxes_on_image(img_path, result)
        st.image(annotated, caption=img_name, use_column_width=True)
        st.write(f"Image {idx+1}/{len(images)}")
        col1, col2 = st.columns(2)
        if col1.button("Đúng", key=f"correct_{idx}"):
            lbl_path = os.path.join(temp_labels_dir, img_name.rsplit('.',1)[0] + '.txt')
            with open(lbl_path, 'a') as f:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    xc, yc, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            st.success(f"Saved labels for {img_name}")
            st.session_state['idx'] += 1
            st.experimental_rerun()
        if col2.button("Sai", key=f"wrong_{idx}"):
            st.info(f"Skipped {img_name}")
            st.session_state['idx'] += 1
            st.experimental_rerun()
    else:
        st.balloons()
        st.success("Annotation completed.")
        # Generate data.yaml
        data_yaml = os.path.join(project_dir, run_name, 'data.yaml')
        os.makedirs(os.path.dirname(data_yaml), exist_ok=True)
        with open(data_yaml, 'w') as f:
            f.write(f"""
train: {IMG_DIR}
val: {IMG_DIR}
nc: 1
names: ['BIB']
""")
        st.write(f"Generated data.yaml at {data_yaml}")
        if st.button("Bắt đầu huấn luyện"):
            with st.spinner("Training in progress..."):
                st.session_state['model'].train(
                    data=data_yaml,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    project=project_dir,
                    name=run_name,
                    exist_ok=True
                )
            st.success("Training completed!")
            # Download button for trained model
            weights_path = os.path.join(project_dir, run_name, 'weights', 'best.pt')
            if os.path.isfile(weights_path):
                with open(weights_path, 'rb') as wf:
                    st.download_button(
                        label="Download fine-tuned model",
                        data=wf,
                        file_name=os.path.basename(weights_path),
                        mime='application/octet-stream'
                    )
            else:
                st.error(f"Weights not found at {weights_path}")
else:
    st.info("Nhấn 'Load Model & Images' để bắt đầu annotation.")

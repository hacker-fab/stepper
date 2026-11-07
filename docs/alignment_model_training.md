# Finetuning a YOLO Model for Alignment

This guide details the process of collecting training images, annotating alignment markers, and training a custom YOLOv11 model with Ultralytics for real-time stepper alignment.

---

## 1. Dataset Collection

To train an alignment detection model, you need a diverse set of chip surface images taken through the stepper's optical path.

### Capturing Images with the Stepper GUI
1. Mount a wafer or test chip with alignment markers onto the stage.
2. In the Camera View panel of the GUI, use the **Snapshot** button to save full-resolution frames of your alignment markers under varied lighting, focus levels, and stage angles.
3. Snapshots are automatically saved to the `stepper_captures/` folder with timestamped filenames.
4. Collect images representing:
   - Varied focus levels (sharp focus, slightly out-of-focus).
   - Different chip substrates (e.g., silicon, glass, oxide layers).
   - Partially degraded or obscured markers.

---

## 2. Data Annotation (Roboflow)

Once you have gathered a set of raw images (typically 50–200 images for a robust fine-tuned detector):

1. Go to [app.roboflow.com](https://app.roboflow.com) and create or sign in to your account.
2. **Create a New Project**:
   - Project Type: **Object Detection**.
   - Target Object: `alignment-markers`.
3. **Upload Data**:
   - Upload your collected images from `stepper_captures/`.
4. **Label Bounding Boxes**:
   - Draw tight bounding boxes around all alignment markers in each image.
   - Use a single consistent class label (e.g., `alignment-marker`).
   - Ensure you assign images across both the **Train** and **Valid** splits.
5. **Preprocessing & Augmentations**:
   - Resize images to 640x640 (standard YOLO input resolution).
   - Recommended augmentations:
     - Grayscale: ~5% of images
     - Brightness adjustments: between -15% and +15%
     - Gaussian noise: up to ~1.5% of pixels
6. **Generate & Export Dataset**:
   - Generate a new dataset version.
   - Choose **Export Dataset** -> format **YOLOv11** -> download as a `.zip` archive.

---

## 3. Training with Ultralytics

### Environment Setup
Install the optional alignment dependencies:
```bash
pip install '.[alignment]'
# or: uv pip install ultralytics
```

Verify that PyTorch and CUDA (if using a dedicated GPU) are functioning properly:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Prepare Dataset Configuration
1. Unzip the downloaded dataset into a working directory.
2. Open the extracted `data.yaml` file.
3. Update the `train` and `val` paths to be absolute paths to avoid path resolution errors during training:
   ```yaml
   train: /path/to/extracted/dataset/train/images
   val: /path/to/extracted/dataset/valid/images
   nc: 1
   names: ['alignment-marker']
   ```

### Run Model Training
Execute the training command using the lightweight YOLOv11n architecture:

```bash
yolo detect train data=/path/to/dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=640
```

Training outputs and evaluation metrics will be written to `runs/detect/train/`.

---

## 4. Deploying Weights to the Stepper

Upon completion of training:
1. Locate the best performing weights at `runs/detect/train/weights/best.pt`.
2. Copy the checkpoint into the stepper software repository:
   ```bash
   cp runs/detect/train/weights/best.pt ckpts/best.pt
   ```
3. Verify your `config.toml` alignment section points to the checkpoint:
   ```toml
   [alignment]
   enabled = true
   model_path = "ckpts/best.pt"
   ```
4. Start the GUI to test real-time detection on the live camera view.

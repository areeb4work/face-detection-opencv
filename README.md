# Face Detection using OpenCV in Python

Internship project (Hex Softwares), a Python program that detects human faces in an image using OpenCV, and draws bounding boxes around each detected face.

## Overview

This project explores two different face detection approaches available in OpenCV:

1. **Haar Cascade Classifier** a classic, fast pattern-matching method built into OpenCV. It works well on clear, front-facing faces but can produce false positives (e.g. mistaking high-contrast textures like pillars for faces) and struggles with angled or partially visible faces.
2. **DNN-based Detector (Deep Learning)** uses a pre-trained Caffe deep learning model (`res10_300x300_ssd_iter_140000`) for more robust detection. It assigns a confidence score to each detection, allowing for more control over precision vs. recall.

The final script uses the DNN-based approach, as it aligns with the deep-learning-based detection method specified in the project brief.

## How It Works

1. Load the input image
2. Prepare it for the neural network (resize + normalize)
3. Run the image through the pre-trained face detection model
4. Keep only detections above a confidence threshold
5. Draw a green bounding box around each detected face
6. Save and output the result as a new image

## Files

| File | Description |
|---|---|
| `detect_faces.py` | Main script, loads the model, runs detection, saves output |
| `deploy.prototxt` | Defines the neural network architecture |
| `res10_300x300_ssd_iter_140000.caffemodel` | Pre-trained weights for the face detection model |
| `face.png` | Sample input image used for testing |
| `face_detected.png` | Output image with bounding boxes drawn around detected faces |

## Requirements

- Python 3.11+
- OpenCV (`opencv-python`)

Install dependencies:
```bash
pip install opencv-python
```

## Usage

1. Place an input image in the project folder and name it `face.png` (or update the filename in the script)
2. Run:
```bash
python detect_faces.py
```
3. The output will be saved as `face_detected.png` in the same folder, and the number of detected faces will be printed to the console

## Example Output

The script prints the number of faces detected and saves an annotated copy of the image:
```
Number of faces detected: 2
Done! Check 'face_detected.png' in this folder.
```

## Notes & Limitations

- Detection accuracy depends on image clarity, face angle, and lighting
- The confidence threshold (`confidence_threshold` in the script) can be tuned lowering it detects more faces but may increase false positives; raising it does the opposite
- Faces that are heavily blurred, angled away from the camera, or partially obstructed may not be detected

## Applications

Face detection techniques like this are widely used in surveillance systems, photo tagging, and user authentication systems.

## Author

Areeb Ahsan - Internship Project for HexSoftwares

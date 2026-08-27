# Image Caption Generator

A production-oriented **Image Captioning System** built with **PyTorch**, **ResNet-50**, **LSTM**, **Streamlit**, and **Docker**.

The system takes an image as input and automatically generates a natural-language caption describing its content.

---

## Project Overview

This project was developed as part of **MIA AI Training 2027 – Phase 2 – Task 6 – Task 1.2: From Notebook to Production**.

The project transforms an image-captioning deep learning experiment into a reusable and deployable application.

### Main pipeline

```text
Input Image
    ↓
ResNet-50
    ↓
2048-D Image Features
    ↓
LSTM Caption Generator
    ↓
Generated Caption
```

---

## Features

- Flickr8k image-caption dataset
- Five human-written reference captions per image
- Train / validation / test split by image
- Caption cleaning and preprocessing
- Vocabulary construction with special tokens
- `<start>`, `<end>`, `<pad>`, and `<unk>` tokens
- Pretrained ResNet-50 for image feature extraction
- Cached image features for faster training
- LSTM-based sequence generation
- Teacher forcing during training
- AdamW optimizer
- Learning-rate scheduling
- Gradient clipping
- Early stopping
- Best-model checkpointing
- BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4 evaluation
- ROUGE-L evaluation
- METEOR evaluation
- Qualitative evaluation with reference captions
- Modular Python source code
- Automated tests with pytest
- Streamlit web interface
- Dockerized deployment

---

##  Project Structure

```text
Task 1.2/
│
├── app.py
├── run_inference.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── Task1_2.ipynb
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── inference.py
│
├── models/
│   ├── best_caption_model.pth
│   ├── resnet50.pth
│   ├── vocabulary.pkl
│   └── config.json
│
├── tests/
│   └── test_inference.py
│
├── Images/
│   └── Flickr8k images
│
└── captions.txt/
    └── captions.txt
```

> Large datasets and model artifacts may be excluded from GitHub and shared through dedicated storage such as Hugging Face.

---

# Dataset

The project uses the **Flickr8k Dataset**.

It contains approximately **8,000 images**, with **five human-written captions associated with each image**.

### Dataset preprocessing

The following steps were applied:

1. Load the image-caption pairs.
2. Verify that caption image names match the images available on disk.
3. Analyze caption lengths and vocabulary.
4. Split the dataset by **unique image names** into:
   - 80% training
   - 10% validation
   - 10% testing
5. Clean captions by:
   - converting text to lowercase
   - removing punctuation
   - removing extra spaces
6. Add special tokens:
   - `<start>`
   - `<end>`
7. Build the vocabulary using the training captions.
8. Replace rare/unknown words with `<unk>`.
9. Convert captions into token IDs.
10. Pad sequences to a fixed maximum length.

### Data leakage prevention

The train, validation, and test sets are created from **unique image names**, not individual captions.

This prevents captions belonging to the same image from appearing in different splits.

---

# 🧠 Model Architecture

## 1. Image Feature Extraction

A pretrained **ResNet-50** model is used as the visual feature extractor.

The final classification layer is removed, producing a:

```text
2048-dimensional image feature vector
```

The extracted features are cached to avoid repeatedly running ResNet-50 during caption-model training.

---

## 2. Caption Generation

The caption generator uses an LSTM architecture.

```text
Image Features (2048)
        │
        ├──────────────→ Initial Hidden State
        │
        └──────────────→ Initial Cell State
                              │
Caption Tokens → Embedding → LSTM → Fully Connected Layer
                                      │
                                      ↓
                              Vocabulary Prediction
```

### Model configuration

| Parameter | Value |
|---|---:|
| Image feature dimension | 2048 |
| Word embedding dimension | 256 |
| LSTM hidden dimension | 512 |
| LSTM layers | 1 |
| Dropout | 0.3 |
| Architecture | ResNet-50 + LSTM |

During inference, captions are generated one token at a time using **greedy decoding** until `<end>` is generated or the maximum caption length is reached.

---

# Training

The model was trained using:

- **Loss:** Cross Entropy Loss
- **Padding:** ignored in the loss calculation
- **Optimizer:** AdamW
- **Initial learning rate:** `1e-3`
- **Weight decay:** `1e-4`
- **Gradient clipping:** maximum norm `5.0`
- **Scheduler:** ReduceLROnPlateau
- **Early stopping:** patience of 3 epochs
- **Checkpointing:** latest and best validation models

The best model is selected according to the lowest validation loss.

---

# Evaluation

The trained model is evaluated on previously unseen test images.

The following image-captioning metrics are used:

- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4
- ROUGE-L
- METEOR

The project also includes qualitative evaluation:

```text
Input Image
     ↓
Generated Caption
     ↓
Five Reference Captions
```

The final metric values are stored in:

```text
models/evaluation_results.json
```

Update the table below with the actual values produced by the final notebook run:

| Metric | Score |
|---|---:|
| BLEU-1 | 0.5862 |
| BLEU-2 | 0.4071 |
| BLEU-3 | 0.2710 |
| BLEU-4 | 0.1798 |
| ROUGE-L | 0.4562 |
| METEOR | 0.3687 |

---

# Local Installation

## Requirements

- Python 3.12
- PyTorch
- Torchvision
- Streamlit
- Pillow
- NumPy
- pytest

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Inference

The production inference pipeline is implemented in:

```text
src/inference.py
```

A command-line inference script is also provided:

```bash
python run_inference.py
```

---

# Run the Streamlit Application

Start the application with:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

Upload an image and click **Generate Caption**.

---

# Docker Deployment

The application is containerized using Docker.

## Build the image

```bash
docker build -t image-caption-generator .
```

## Run the container

```bash
docker run -d -p 8501:8501 --name caption-app image-caption-generator
```

The application will be available at:

```text
http://localhost:8501
```

## Check the container

```bash
docker ps
```

## View application logs

```bash
docker logs caption-app
```

To stop the application:

```bash
docker stop caption-app
```

To remove the container:

```bash
docker rm caption-app
```

> If the container name `caption-app` is already in use, do not run `docker run` again. Use `docker logs caption-app` or `docker start caption-app`.

---

# Testing

The project includes automated tests using `pytest`.

Run:

```bash
pytest
```

The tests verify:

- Required model artifacts exist.
- The caption-generation pipeline returns a non-empty string for a test image.

---

# Model Storage

The trained model artifacts should be shared through a public model repository such as **Hugging Face Hub**.

Required production artifacts:

```text
best_caption_model.pth
resnet50.pth
vocabulary.pkl
config.json
```

### Hugging Face Model

**Public Model Link:**  
`TODO: ADD HUGGING FACE MODEL LINK`

---

# Public Demo

The Streamlit application can be deployed to a public hosting service.

**Live Demo:**  
`TODO: ADD PUBLIC STREAMLIT / HOSTING LINK`

---

# Project Links

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/salmaa203/Image-Caption-Generator.git |
| Hugging Face Model | https://huggingface.co/salmaelshehy/image-caption-generator |
| Live Demo | https://image-caption-generator-bbuxl5sxt3wognbu4r6kkr.streamlit.app/ |
---

# Demo

Add a short screen recording showing:

1. Opening the application.
2. Uploading an image.
3. Generating a caption.
4. Displaying the generated result.

Recommended location:

```text
README.md
```

For example:

```markdown
## Demo

[Watch the demo video](YOUR_VIDEO_LINK)
```

---

# Limitations

The model can identify general activities and scene context, but it may occasionally generate:

- incorrect object attributes
- incorrect relationships between objects
- repetitive phrases
- captions that are grammatically valid but semantically imperfect

These limitations are expected from a relatively lightweight **LSTM-based caption-generation architecture** trained on the Flickr8k dataset.

---

# Possible Future Improvements

- Attention mechanism
- Beam search instead of greedy decoding
- Transformer-based decoder
- Larger pretrained vision encoder
- Fine-tuning the CNN
- Better handling of rare words
- More advanced captioning metrics
- Larger image-caption datasets
- Model quantization for faster deployment

---

# Technologies

- Python
- PyTorch
- Torchvision
- ResNet-50
- LSTM
- NLTK
- ROUGE
- Streamlit
- Docker
- pytest
- Hugging Face Hub

---

# Author

**Salma Elshehy**

MIA AI Training 2027  
Phase 2 – Task 6 – Task 1.2

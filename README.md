# Image Caption Generator

A production-oriented **Image Captioning System** built with **PyTorch, ResNet-50, LSTM, Streamlit, Docker, and Hugging Face Hub**.

The system takes an image as input and automatically generates a natural-language caption describing its content.

---

## Project Overview

This project was developed as part of:

**MIA AI Training 2027 – Phase 2 – Task 6 – Task 1.2: From Notebook to Production**

The project transforms an image-captioning deep learning experiment into a reusable, modular, tested, and deployable application.

### Main Pipeline

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
- Train / validation / test split by unique images
- Caption cleaning and preprocessing
- Vocabulary construction
- Special tokens: `<start>`, `<end>`, `<pad>`, and `<unk>`
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
- Automated tests using pytest
- Streamlit web interface
- Dockerized application
- Hugging Face model storage
- Public Streamlit deployment

---

# Project Structure

```text
Image-Caption-Generator/
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
└── captions.txt
```

> Large datasets and model artifacts are stored separately and shared through Hugging Face Hub.

---

# Dataset

The project uses the **Flickr8k Dataset**, which contains approximately **8,000 images**, with **five human-written captions associated with each image**.

## Dataset Preprocessing

The following preprocessing steps were applied:

1. Load the image-caption pairs.
2. Verify that caption image names match the available images.
3. Analyze caption lengths and vocabulary.
4. Split the dataset by **unique image names** into:
   - 80% training
   - 10% validation
   - 10% testing
5. Convert captions to lowercase.
6. Remove punctuation and extra spaces.
7. Add special tokens:
   - `<start>`
   - `<end>`
8. Build the vocabulary using training captions.
9. Replace rare or unknown words with `<unk>`.
10. Convert captions into numerical token IDs.
11. Pad sequences to a fixed maximum length.

## Data Leakage Prevention

The dataset is split using **unique image names rather than individual captions**.

Since every image has five captions, splitting individual captions could cause captions belonging to the same image to appear in both training and testing sets.

Splitting by image prevents this form of data leakage and ensures that test images are unseen during training.

---

# Model Architecture

## 1. Image Feature Extraction

A pretrained **ResNet-50** model is used as the visual feature extractor.

The final classification layer is removed, producing a:

```text
2048-dimensional image feature vector
```

The extracted image features are cached to avoid repeatedly running ResNet-50 during caption-model training.

This significantly improves training efficiency.

---

## 2. Caption Generation

The caption generator uses an **LSTM-based sequence model**.

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

### Model Configuration

| Parameter | Value |
|---|---:|
| Image feature dimension | 2048 |
| Word embedding dimension | 256 |
| LSTM hidden dimension | 512 |
| LSTM layers | 1 |
| Dropout | 0.3 |
| Architecture | ResNet-50 + LSTM |

During inference, the model generates the caption one word at a time using **greedy decoding**.

Generation stops when the `<end>` token is produced or the maximum caption length is reached.

---

# Training

The model was trained using:

| Component | Configuration |
|---|---|
| Loss | Cross Entropy Loss |
| Optimizer | AdamW |
| Initial Learning Rate | `1e-3` |
| Weight Decay | `1e-4` |
| Gradient Clipping | Maximum norm `5.0` |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | Patience = 3 epochs |
| Checkpointing | Best validation model |

Padding tokens are ignored when calculating the training loss.

The best model is selected according to the lowest validation loss.

---

# Evaluation

The final model was evaluated on previously unseen test images.

The following image-captioning metrics were used:

- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4
- ROUGE-L
- METEOR

## Final Test Results

| Metric | Score |
|---|---:|
| BLEU-1 | 0.5862 |
| BLEU-2 | 0.4071 |
| BLEU-3 | 0.2710 |
| BLEU-4 | 0.1798 |
| ROUGE-L | 0.4562 |
| METEOR | 0.3687 |

The evaluation results are also stored in:

```text
evaluation_results.json
```

---

# Qualitative Evaluation

The project also includes qualitative evaluation by comparing generated captions with the five human-written reference captions associated with each test image.

```text
Input Image
     ↓
Generated Caption
     ↓
Five Reference Captions
```

This provides a more interpretable evaluation of the model's ability to describe image content.

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

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Inference

The production inference pipeline is implemented in:

```text
src/inference.py
```

To run inference from the command line:

```bash
python run_inference.py
```

---

# Run the Streamlit Application

Start the application locally:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

Upload an image and click:

**Generate Caption**

The application will process the image using ResNet-50 and generate a caption using the trained LSTM model.

---

# Docker Deployment

The application is containerized using Docker.

## Build the Docker Image

```bash
docker build -t image-caption-generator .
```

## Run the Container

```bash
docker run -d -p 8501:8501 --name caption-app image-caption-generator
```

The application will be available at:

```text
http://localhost:8501
```

## Check the Running Container

```bash
docker ps
```

## View Application Logs

```bash
docker logs caption-app
```

## Stop the Container

```bash
docker stop caption-app
```

## Remove the Container

```bash
docker rm caption-app
```

> If the container name `caption-app` is already in use, do not run `docker run` again. Use `docker start caption-app` or inspect the logs with `docker logs caption-app`.

---

# Testing

The project includes automated tests using **pytest**.

Run:

```bash
pytest
```

The tests verify:

- Required model artifacts are available.
- The inference pipeline can load the model.
- The caption-generation pipeline returns a non-empty string for a test image.

---

# Model Storage and Sharing

The trained production artifacts are stored in a public **Hugging Face Hub** repository.

Required artifacts include:

```text
best_caption_model.pth
resnet50.pth
vocabulary.pkl
config.json
```

These files are downloaded automatically by the production inference pipeline using `huggingface_hub`.

This allows the application to load the model without storing large model files directly inside the GitHub repository.

### Hugging Face Model

**Model Repository:**

https://huggingface.co/salmaelshehy/image-caption-generator

---

# Public Demo

The application is deployed as a public Streamlit application.

### Live Demo

https://image-caption-generator-bbuxl5sxt3wognbu4r6kkr.streamlit.app/

Users can upload an image and generate a caption directly through the web interface.

---

# Demo Video

A short demonstration video shows:

1. Opening the application.
2. Uploading an image.
3. Generating a caption.
4. Displaying the generated caption.

### Watch the Demo

https://drive.google.com/file/d/1tEg2pyKOtYzQZMlAuQorjRYawoh0AqJG/view?usp=sharing

---

# Project Links

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/salmaa203/Image-Caption-Generator.git |
| Hugging Face Model | https://huggingface.co/salmaelshehy/image-caption-generator |
| Live Demo | https://image-caption-generator-bbuxl5sxt3wognbu4r6kkr.streamlit.app/ |
| Demo Video | https://drive.google.com/file/d/1tEg2pyKOtYzQZMlAuQorjRYawoh0AqJG/view?usp=sharing |

---

# Limitations

Although the model can identify general objects, activities, and scene context, it may occasionally generate:

- Incorrect object attributes.
- Incorrect relationships between objects.
- Repetitive phrases.
- Grammatically valid but semantically imperfect captions.

These limitations are expected from a relatively lightweight **LSTM-based caption-generation architecture** trained on the Flickr8k dataset.

---

# Future Improvements

Possible improvements include:

- Attention mechanism.
- Beam search instead of greedy decoding.
- Transformer-based decoder.
- Larger pretrained vision encoder.
- Fine-tuning the CNN feature extractor.
- Better handling of rare words.
- More advanced captioning metrics.
- Training on larger image-caption datasets.
- Model quantization for faster inference.

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
Phase 2 – Task 6 – Task 1.2: From Notebook to Production

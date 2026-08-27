import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from huggingface_hub import hf_hub_download

from .model import ImageCaptioningModel


# ============================================================
# Hugging Face Repository
# ============================================================

HF_REPO_ID = "salmaelshehy/image-caption-generator"


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Download Model Artifacts from Hugging Face
# ============================================================

def download_artifact(filename):

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename
    )


MODEL_PATH = download_artifact(
    "best_caption_model.pth"
)

RESNET_PATH = download_artifact(
    "resnet50.pth"
)

VOCAB_PATH = download_artifact(
    "vocabulary.pkl"
)

CONFIG_PATH = download_artifact(
    "config.json"
)


# ============================================================
# Load Configuration
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:

    config = json.load(f)


# ============================================================
# Load Vocabulary
# ============================================================

with open(
    VOCAB_PATH,
    "rb"
) as f:

    vocabulary = pickle.load(f)


word2idx = vocabulary["word2idx"]
idx2word = vocabulary["idx2word"]


# ============================================================
# Load Caption Model
# ============================================================

caption_model = ImageCaptioningModel(
    feature_dim=config["feature_dim"],
    vocab_size=config["vocab_size"],
    embed_dim=256,
    hidden_dim=512,
    num_layers=1,
    dropout=0.3,
    pad_idx=config["pad_idx"]
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)


caption_model.load_state_dict(
    checkpoint["model_state_dict"]
)

caption_model = caption_model.to(DEVICE)

caption_model.eval()

# ============================================================
# Load ResNet-50 Feature Extractor
# ============================================================

base_resnet = models.resnet50(
    weights=None
)

resnet = nn.Sequential(
    *list(base_resnet.children())[:-1]
)

resnet.load_state_dict(
    torch.load(
        RESNET_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

resnet = resnet.to(DEVICE)

resnet.eval()
# ============================================================
# Image Preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# Extract Image Features
# ============================================================

def extract_features(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        features = resnet(
            image_tensor
        )

    features = features.squeeze(
        -1
    ).squeeze(
        -1
    )

    return features.squeeze(0)


# ============================================================
# Generate Caption
# ============================================================

def generate_caption(image_path):

    feature = extract_features(
        image_path
    )

    feature = feature.unsqueeze(0)

    feature = feature.to(DEVICE)

    with torch.no_grad():

        # Initialize hidden state
        h = caption_model.init_h(
            feature
        )

        # Initialize cell state
        c = caption_model.init_c(
            feature
        )

        h = h.unsqueeze(0)

        c = c.unsqueeze(0)

        # Start token
        current_word = torch.tensor(
            [[config["start_idx"]]],
            dtype=torch.long,
            device=DEVICE
        )

        generated_words = []

        # Generate one word at a time
        for _ in range(
            config["max_len"]
        ):

            # Word embedding
            embedding = caption_model.embedding(
                current_word
            )

            # LSTM step
            output, (h, c) = caption_model.lstm(
                embedding,
                (h, c)
            )

            # Vocabulary prediction
            output = caption_model.fc(
                output
            )

            # Greedy decoding
            predicted_id = output.argmax(
                dim=-1
            ).item()

            # Stop at <end>
            if predicted_id == config["end_idx"]:
                break

            # Ignore padding
            if predicted_id != config["pad_idx"]:

                word = idx2word[
                    predicted_id
                ]

                # Ignore special tokens
                if word not in [
                    "<start>",
                    "<end>",
                    "<pad>"
                ]:

                    generated_words.append(
                        word
                    )

            # Next input word
            current_word = torch.tensor(
                [[predicted_id]],
                dtype=torch.long,
                device=DEVICE
            )

    return " ".join(
        generated_words
    )

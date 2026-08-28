from src.inference import generate_caption


# Test Image

IMAGE_PATH = "Images/1000268201_693b08cb0e.jpg"


# Generate Caption

print("=" * 60)
print("PRODUCTION INFERENCE TEST")
print("=" * 60)

print(f"Image: {IMAGE_PATH}")

caption = generate_caption(
    IMAGE_PATH
)

print(f"Generated Caption: {caption}")

print("=" * 60)

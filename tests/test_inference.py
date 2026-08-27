from pathlib import Path

from src.inference import generate_caption


def test_model_files_exist():

    base_dir = Path(
        __file__
    ).resolve().parent.parent

    model_dir = base_dir / "models"

    required_files = [
        "best_caption_model.pth",
        "resnet50.pth",
        "vocabulary.pkl",
        "config.json",
    ]

    for filename in required_files:

        path = model_dir / filename

        assert path.exists(), (
            f"Missing model artifact: {filename}"
        )


def test_caption_generation():

    base_dir = Path(
        __file__
    ).resolve().parent.parent

    image_path = (
        base_dir
        / "Images"
        / "1000268201_693b08cb0e.jpg"
    )

    assert image_path.exists(), (
        f"Test image not found: {image_path}"
    )

    caption = generate_caption(
        str(image_path)
    )

    assert isinstance(
        caption,
        str
    )

    assert len(
        caption.strip()
    ) > 0
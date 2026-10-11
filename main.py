import torch
from src.trainer import get_or_train_model
from src.gui import run_app


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pre-trained weights or run training loop
    model = get_or_train_model(model_path="mnist_model.pth", device=device)

    # Launch GUI
    run_app(model, device)


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import pygame
import torch
import torch.nn as nn


def process_drawing(surface_canvas: pygame.Surface, device: str):
    surface = pygame.surfarray.array3d(surface_canvas)
    gray = np.dot(surface[..., :3], [0.2989, 0.5870, 0.1140])
    gray = np.transpose(gray, (1, 0)).astype(np.uint8)

    coords = cv2.findNonZero(gray)
    if coords is None:
        return None, None

    x, y, w, h = cv2.boundingRect(coords)
    digit_crop = gray[y:y+h, x:x+w]

    max_dim = max(w, h)
    scale = 20.0 / max_dim
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    padded_digit = np.zeros((28, 28), dtype=np.uint8)
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    padded_digit[start_y:start_y+new_h, start_x:start_x+new_w] = resized_digit

    padded_digit = cv2.GaussianBlur(padded_digit, (3, 3), 0)

    normalized = padded_digit.astype(np.float32) / 255.0
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor.to(device), padded_digit


def predict_digit(model: nn.Module, surface_canvas: pygame.Surface, device: str):
    tensor, raw_28x28 = process_drawing(surface_canvas, device)
    if tensor is None:
        return None, None

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(dim=1, keepdim=True)

    return pred.item(), raw_28x28
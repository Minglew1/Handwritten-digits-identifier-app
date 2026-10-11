import numpy as np
import pygame
import torch.nn as nn

from src.processing import predict_digit


def run_app(model: nn.Module, device: str):
    pygame.init()
    canvas_size = 280
    display_height = canvas_size + 70
    screen = pygame.display.set_mode((canvas_size, display_height))
    pygame.display.set_caption("Digit Recognizer")
    clock = pygame.time.Clock()

    canvas = pygame.Surface((canvas_size, canvas_size))
    canvas.fill((0, 0, 0))

    drawing = False
    prediction = None
    preview_28x28 = None
    font = pygame.font.Font(None, 36)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.pos[1] < canvas_size:
                    drawing = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if drawing:
                    drawing = False
                    prediction, preview_28x28 = predict_digit(model, canvas, device)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    canvas.fill((0, 0, 0))
                    prediction = None
                    preview_28x28 = None

            elif event.type == pygame.MOUSEMOTION and drawing:
                if event.pos[1] < canvas_size:
                    pygame.draw.circle(canvas, (255, 255, 255), event.pos, 15)

        screen.fill((30, 30, 30))
        screen.blit(canvas, (0, 0))

        if prediction is not None and preview_28x28 is not None:
            preview_rgb = np.stack([preview_28x28] * 3, axis=-1)
            preview_rgb = np.transpose(preview_rgb, (1, 0, 2))

            surf_28x28 = pygame.surfarray.make_surface(preview_rgb)
            scaled_preview = pygame.transform.scale(surf_28x28, (56, 56))

            screen.blit(scaled_preview, (canvas_size - 66, canvas_size + 7))
            pygame.draw.rect(screen, (250, 250, 250), (canvas_size - 67, canvas_size + 6, 58, 58), 1)

            text = font.render(f"Pred: {prediction}", True, (0, 255, 0))
            screen.blit(text, (10, canvas_size + 20))
        else:
            text = font.render("Draw & release mouse", True, (150, 150, 150))
            screen.blit(text, (10, canvas_size + 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
import io
from PIL import Image, ImageDraw
import random

from matplotlib import pyplot as plt
import numpy as np


def generate_image_with_circles(width=300, height=300, max_circles=10):
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    num_circles = random.randint(1, max_circles)
    for _ in range(num_circles):
        x = random.randint(20, width - 20)
        y = random.randint(20, height - 20)
        radius = random.randint(10, 30)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=tuple(random.randint(0, 255) for _ in range(3)),
            outline="black",
        )
    draw.rectangle([0.5, 0.5, width - 1, height - 1], outline="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Image.open(buffer), num_circles


def generate_image_with_lines(num_points=3):
    def find_intersections(x, y1, y2):
        intersections = []
        for i in range(len(x) - 1):
            if (y1[i] - y2[i]) * (y1[i + 1] - y2[i + 1]) < 0:
                intersections.append((x[i] + x[i + 1]) / 2)
        return len(intersections)

    x = np.linspace(0, 10, num_points)
    y1 = np.random.rand(num_points) * 10
    y2 = np.random.rand(num_points) * 10
    num_intersections = find_intersections(x, y1, y2)
    line_thickness = 4  # Consider 2 or 3
    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, "r-", linewidth=line_thickness)
    plt.plot(x, y2, "b-", linewidth=line_thickness)
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return Image.open(buf), num_intersections


def generate_random_scatter_plot(n=500, size=(8, 8), dpi=100):
    fig, ax = plt.subplots(figsize=size)
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    ax.scatter(x, y, s=10, alpha=0.7, edgecolors="none")
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

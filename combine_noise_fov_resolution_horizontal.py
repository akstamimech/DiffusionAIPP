"""
Horizontally concatenates the three independent noise/FOV/resolution panels
(same panels combine_noise_fov_resolution.py stacks vertically) side by side
for a slide layout:
  - sensor_noise_model.png                  (plot_sensor_noise_model.py)
  - fov_vs_altitude_panel_map11_NAIP.png     (plot_fov_resolution_panels_separate.py)
  - resolution_blocks_panel_map11_NAIP.png   (plot_fov_resolution_panels_separate.py)

Run plot_sensor_noise_model.py and plot_fov_resolution_panels_separate.py
first to (re)generate these three PNGs before running this.
"""
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
NOISE_PATH = SCRIPT_DIR / "sensor_noise_model.png"
FOV_PATH = SCRIPT_DIR / "fov_vs_altitude_panel_map11_NAIP.png"
RESOLUTION_PATH = SCRIPT_DIR / "resolution_blocks_panel_map11_NAIP.png"
OUT_PATH = SCRIPT_DIR / "sensor_fov_resolution_horizontal.png"

SURFACE = (252, 252, 251)  # matches the SURFACE hex (#fcfcfb) used in the source figures
GAP_PX = 32
TARGET_HEIGHT = 720


def scale_to_height(img, height):
    if img.height == height:
        return img
    new_width = round(img.width * (height / img.height))
    return img.resize((new_width, height), Image.LANCZOS)


def main():
    panels = [
        Image.open(NOISE_PATH).convert("RGB"),
        Image.open(FOV_PATH).convert("RGB"),
        Image.open(RESOLUTION_PATH).convert("RGB"),
    ]
    panels = [scale_to_height(p, TARGET_HEIGHT) for p in panels]

    total_width = sum(p.width for p in panels) + GAP_PX * (len(panels) - 1)
    combined = Image.new("RGB", (total_width, TARGET_HEIGHT), SURFACE)

    x = 0
    for p in panels:
        combined.paste(p, (x, 0))
        x += p.width + GAP_PX

    combined.save(OUT_PATH)
    print(f"Wrote {OUT_PATH} ({combined.width}x{combined.height})")


if __name__ == "__main__":
    main()

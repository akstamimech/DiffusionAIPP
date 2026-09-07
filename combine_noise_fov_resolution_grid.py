"""
Same three independent panels as combine_noise_fov_resolution_horizontal.py,
arranged in a squarer 2-row grid instead of one wide strip - meant to fit a
slide better:
  Row 1: sensor_noise_model.png + fov_vs_altitude_panel_map11_NAIP.png
  Row 2: resolution_blocks_panel_map11_NAIP.png, stretched to the row-1 width

Run plot_sensor_noise_model.py and plot_fov_resolution_panels_separate.py
first to (re)generate these three PNGs before running this.
"""
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
NOISE_PATH = SCRIPT_DIR / "sensor_noise_model.png"
FOV_PATH = SCRIPT_DIR / "fov_vs_altitude_panel_map11_NAIP.png"
RESOLUTION_PATH = SCRIPT_DIR / "resolution_blocks_panel_map11_NAIP.png"
OUT_PATH = SCRIPT_DIR / "sensor_fov_resolution_grid.png"

SURFACE = (252, 252, 251)  # matches the SURFACE hex (#fcfcfb) used in the source figures
GAP_PX = 32
ROW1_HEIGHT = 500


def scale_to_height(img, height):
    if img.height == height:
        return img
    new_width = round(img.width * (height / img.height))
    return img.resize((new_width, height), Image.LANCZOS)


def scale_to_width(img, width):
    if img.width == width:
        return img
    new_height = round(img.height * (width / img.width))
    return img.resize((width, new_height), Image.LANCZOS)


def main():
    noise = scale_to_height(Image.open(NOISE_PATH).convert("RGB"), ROW1_HEIGHT)
    fov = scale_to_height(Image.open(FOV_PATH).convert("RGB"), ROW1_HEIGHT)
    row1_width = noise.width + GAP_PX + fov.width

    resolution = scale_to_width(Image.open(RESOLUTION_PATH).convert("RGB"), row1_width)

    total_width = row1_width
    total_height = ROW1_HEIGHT + GAP_PX + resolution.height
    combined = Image.new("RGB", (total_width, total_height), SURFACE)

    combined.paste(noise, (0, 0))
    combined.paste(fov, (noise.width + GAP_PX, 0))
    combined.paste(resolution, (0, ROW1_HEIGHT + GAP_PX))

    combined.save(OUT_PATH)
    print(f"Wrote {OUT_PATH} ({combined.width}x{combined.height}, aspect {combined.width/combined.height:.2f}:1)")


if __name__ == "__main__":
    main()

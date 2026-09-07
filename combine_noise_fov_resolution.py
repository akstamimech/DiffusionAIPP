"""
Vertically concatenates plot_sensor_noise_model.png (top) and
plot_fov_resolution_altitude.py's output (bottom) into one combined figure for
the thesis/IEEE paper - same PIL-vertical-concatenation approach already used
in this project for map_200_planner_heatmaps_combined.png.

Run plot_sensor_noise_model.py and plot_fov_resolution_altitude.py first to
(re)generate their PNGs before running this.
"""
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
NOISE_PATH = SCRIPT_DIR / "sensor_noise_model.png"
FOV_PATH = SCRIPT_DIR / "fov_resolution_vs_altitude_map11_NAIP.png"
OUT_PATH = SCRIPT_DIR / "sensor_fov_resolution_combined.png"

SURFACE = (252, 252, 251)  # matches the SURFACE hex (#fcfcfb) used in both source figures
GAP_PX = 24


def main():
    top = Image.open(NOISE_PATH).convert("RGB")
    bottom = Image.open(FOV_PATH).convert("RGB")

    target_width = max(top.width, bottom.width)

    def scale_to_width(img, width):
        if img.width == width:
            return img
        new_height = round(img.height * (width / img.width))
        return img.resize((width, new_height), Image.LANCZOS)

    top = scale_to_width(top, target_width)
    bottom = scale_to_width(bottom, target_width)

    combined = Image.new("RGB", (target_width, top.height + GAP_PX + bottom.height), SURFACE)
    combined.paste(top, (0, 0))
    combined.paste(bottom, (0, top.height + GAP_PX))
    combined.save(OUT_PATH)
    print(f"Wrote {OUT_PATH} ({combined.width}x{combined.height})")


if __name__ == "__main__":
    main()

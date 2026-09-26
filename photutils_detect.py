#!/usr/bin/env python3

"""
Detect astronomical sources in headerless 4096x3000 uint8 RAW frames from
SOBER dataset using Photutils.

Outputs:
    detections.csv       - detected source measurements
    detections.png       - visualisation of detections
    segmentation.png     - segmentation map

Run:
    python photoutils.py 
    python photoutils.py ***frame.raw*** if you want to run on one frame
"""

#!/usr/bin/env python3

import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import detect_sources, SourceCatalog

WIDTH = 4096
HEIGHT = 3000
DTYPE = np.uint8
EXPECTED_BYTES = WIDTH * HEIGHT

def load_raw(path):
    path = Path(path)
    file_size = path.stat().st_size

    if file_size != EXPECTED_BYTES:
        raise ValueError(
            f"{path} is {file_size:,} bytes.\n"
            f"Expected {EXPECTED_BYTES:,} bytes for "
            f"{WIDTH}x{HEIGHT} uint8 data."
        )

    image = np.fromfile(path, dtype=DTYPE)
    return image.reshape((HEIGHT, WIDTH))


def detect_objects(image):
    print("Estimating background and noise...")

    mean, median, std = sigma_clipped_stats(
        image,
        sigma=3.0,
        maxiters=5
    )

    print(f"Background median : {median:.3f}")
    print(f"Background std    : {std:.3f}")

    threshold = median + 5.0 * std
    print(f"Detection threshold: {threshold:.3f}")

    segmentation = detect_sources(
        image,
        threshold,
        n_pixels=5
    )

    if segmentation is None:
        print("No sources detected.")
        return None, None

    print(f"Detected {segmentation.n_labels} sources.")

    background = np.full_like(
        image,
        median,
        dtype=np.float32
    )

    catalog = SourceCatalog(
        image,
        segmentation,
        background=background
    )

    return segmentation, catalog


def save_catalog(catalog, output_path):
    """Save SourceCatalog measurements to CSV."""

    rows = []

    for source in catalog:
        semimajor = float(source.semimajor_axis.to_value("pix"))
        semiminor = float(source.semiminor_axis.to_value("pix"))
        elongation = semimajor / max(semiminor, 1e-9)

        row = {
            "id": int(source.label),
            "x": float(source.x_centroid),
            "y": float(source.y_centroid),
            "x_min": int(source.bbox_xmin),
            "x_max": int(source.bbox_xmax),
            "y_min": int(source.bbox_ymin),
            "y_max": int(source.bbox_ymax),
            "area": float(source.area.to_value("pix2")),
            "semimajor_axis": semimajor,
            "semiminor_axis": semiminor,
            "elongation": elongation,
            "orientation_deg": float(
                source.orientation.to_value("deg")
            ),
            "eccentricity": float(source.eccentricity),
            "segment_flux": float(source.segment_flux),
        }

        rows.append(row)

    if not rows:
        print("No sources to save.")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved catalog: {output_path}")


def save_detection_plot(image, catalog, output_file):
    display = image.astype(float)

    low = np.percentile(display, 1)
    high = np.percentile(display, 99.9)
    display = np.clip(display, low, high)

    plt.figure(figsize=(16, 10))

    plt.imshow(
        display,
        cmap="gray",
        origin="lower"
    )

    if catalog is not None:
        for source in catalog:
            x = float(source.x_centroid)
            y = float(source.y_centroid)

            x0 = source.bbox_xmin
            x1 = source.bbox_xmax
            y0 = source.bbox_ymin
            y1 = source.bbox_ymax

            plt.plot(
                [x0, x1, x1, x0, x0],
                [y0, y0, y1, y1, y0],
                linewidth=1
            )

            plt.plot(
                x,
                y,
                marker="+",
                markersize=8,
                markeredgewidth=1
            )

            plt.text(
                x,
                y,
                str(source.label),
                fontsize=6
            )

    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.title("Photutils source detections")

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved detection image: {output_file}")


def save_segmentation_plot(segmentation, output_file):
    plt.figure(figsize=(16, 10))

    plt.imshow(
        segmentation.data,
        origin="lower",
        interpolation="nearest"
    )

    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")
    plt.title("Photutils segmentation map")

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved segmentation map: {output_file}")


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "    python photutils_detect.py frame.raw"
        )
        sys.exit(1)

    raw_file = Path(sys.argv[1])

    if not raw_file.exists():
        print(f"File not found: {raw_file}")
        sys.exit(1)

    print(f"Loading: {raw_file}")

    image = load_raw(raw_file)

    print(
        f"Loaded image: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    segmentation, catalog = detect_objects(image)

    if catalog is None:
        return

    stem = raw_file.stem

    save_catalog(
        catalog,
        f"{stem}_detections.csv"
    )

    save_detection_plot(
        image,
        catalog,
        f"{stem}_detections.png"
    )

    save_segmentation_plot(
        segmentation,
        f"{stem}_segmentation.png"
    )


if __name__ == "__main__":
    main()


np.linalg.norm
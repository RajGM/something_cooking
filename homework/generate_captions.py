from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import (
    draw_detections,
    extract_frame_info,
    extract_kart_objects,
    extract_track_info,
)


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate captions for a specific view.

    Returns:
        A list of caption strings describing the scene.
        The outer script will pair each caption with the image_file
        to produce ..._captions.json for training CLIP.
    """
    captions: list[str] = []

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return captions

    track_name = extract_track_info(info_path)

    # Ego kart
    ego_kart = next((k for k in karts if k["is_center_kart"]), karts[0])
    ego_name = ego_kart["kart_name"]
    ego_cx, ego_cy = ego_kart["center"]

    def left_right(k):
        cx, _ = k["center"]
        dx = cx - ego_cx
        if abs(dx) < img_width * 0.02:
            return "center"
        return "left" if dx < 0 else "right"

    def front_back(k):
        _, cy = k["center"]
        dy = cy - ego_cy
        if abs(dy) < img_height * 0.02:
            return "same position"
        return "in front of" if dy < 0 else "behind"

    # 1. Ego car
    captions.append(f"{ego_name} is the ego car.")

    # 2. Counting
    captions.append(f"There are {len(karts)} karts in the scenario.")

    # 3. Track name
    captions.append(f"The track is {track_name}.")

    # 4. Relative positions of other karts
    for k in karts:
        if k["instance_id"] == ego_kart["instance_id"]:
            continue

        name = k["kart_name"]
        lr = left_right(k)
        fb = front_back(k)

        if lr == "center" and fb == "same position":
            rel = "at the same position as the ego car"
        elif lr == "center":
            rel = f"{fb} the ego car"
        elif fb == "same position":
            rel = f"to the {lr} of the ego car"
        else:
            rel = f"{fb} and to the {lr} of the ego car"

        captions.append(f"{name} is {rel}.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize captions for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0
"""


def main():
    fire.Fire({"check": check_caption})


if __name__ == "__main__":
    main()

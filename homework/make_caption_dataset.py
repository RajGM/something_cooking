import json
from pathlib import Path
from homework.generate_captions import generate_caption

DATA_ROOT = Path("data/train")
OUTPUT = DATA_ROOT / "generated_captions.json"

def main():
    all_items = []

    for info_file in sorted(DATA_ROOT.glob("*_info.json")):
        base = info_file.stem.replace("_info", "")

        for view in range(6):
            image_path = info_file.parent / f"{base}_{view:02d}_im.jpg"
            if not image_path.exists():
                continue

            captions = generate_caption(str(info_file), view)
            for cap in captions:
                all_items.append({
                    "image_file": f"train/{image_path.name}",
                    "caption": cap,
                })

    print(f"Generated {len(all_items)} captions.")
    with open(OUTPUT, "w") as f:
        json.dump(all_items, f, indent=2)

if __name__ == "__main__":
    main()

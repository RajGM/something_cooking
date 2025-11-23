import json
from pathlib import Path
from homework.generate_qa import generate_qa_pairs

DATA_ROOT = Path("data/train")
OUTPUT = DATA_ROOT / "generated_qa_pairs.json"

def main():
    all_items = []

    for info_file in sorted(DATA_ROOT.glob("*_info.json")):
        base = info_file.stem.replace("_info", "")

        # Try all camera views: 0..5 (usually 6 cameras)
        for view in range(6):
            image_path = info_file.parent / f"{base}_{view:02d}_im.jpg"
            if not image_path.exists():
                continue

            qa_pairs = generate_qa_pairs(str(info_file), view)
            for qa in qa_pairs:
                all_items.append({
                    "image_file": f"train/{image_path.name}",
                    "question": qa["question"],
                    "answer": qa["answer"]
                })

    print(f"Generated {len(all_items)} QA items.")
    with open(OUTPUT, "w") as f:
        json.dump(all_items, f, indent=2)

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import os
from glob import glob
import argparse
import shutil

"""
Label assignments:
1: Plausible asymmetric event
2: Stellar Variability
3: Artefact/Not considered
4: Unsure


"""



argparser = argparse.ArgumentParser(description="Assign categories to exocomet plots from ML output.")

argparser.add_argument(
    "-p",
    "--path",
    type=str,
    dest='path',
    help="Target folder containing light curve plots. Assumes PNG format.",
)

argparser.add_argument(
    "-o",
    "--output",
    type=str,
    dest='output',
    help="Output CSV file for label assignments.",
)

argparser.add_argument(
    "-c",
    "--checkpoint",
    type=str,
    dest='checkpoint',
    help="Checkpoint file to load assignments from.",
)


argparser.add_argument(
    "-m",
    "--move_dir",
    type=str,
    dest='move_dir',
    help="Optional directory to copy plots into, organised by label after finishing.",
)


args = argparser.parse_args()

# Setup
image_folder = args.path
image_files = sorted(glob(os.path.join(image_folder, "*.png")))
current_index = 0
assignments = {}

# Load from checkpoint
if os.path.exists(args.output):
    import pandas as pd
    df = pd.read_csv(args.output)
    assignments = dict(zip(df["filename"], df["label"]))

def show_image(index):
    img = plt.imread(image_files[index])
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{os.path.basename(image_files[index])} | {assignments.get(os.path.basename(image_files[index]), '')}")
    plt.draw()

def on_key(event):
    global current_index
    filename = os.path.basename(image_files[current_index])

    if event.key in ["1", "2", "3", "4"]:
        assignments[filename] = f"category{event.key}"
        print(f"{filename} → category{event.key}")
        current_index += 1
    elif event.key == "right":
        current_index += 1
    elif event.key == "left":
        current_index = max(0, current_index - 1)
    elif event.key == "escape":
        save_and_quit()
        return

    current_index = current_index % len(image_files)
    plt.clf()
    show_image(current_index)

def save_and_quit():
    import pandas as pd
    df = pd.DataFrame([
        {"filename": k, "label": v} for k, v in assignments.items()
    ])
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")

    # Copy files to move_dir if specified
    if args.move_dir:
        for filename, label in assignments.items():
            src_path = os.path.join(image_folder, filename)
            label_folder = os.path.join(args.move_dir, label)
            os.makedirs(label_folder, exist_ok=True)
            dst_path = os.path.join(label_folder, filename)
            shutil.copy(src_path, dst_path)
        print(f"Copied labeled files to {args.move_dir}/<category>/")
    
    plt.close()

# Run
fig = plt.figure()
show_image(current_index)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

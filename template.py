from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%asctime]: %(message)s')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "requirements.txt",
    "app.py",
    "research/trails.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if(not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating file: {filepath}")
    else:
        logging.info(f"{filename} is already exists and not empty.")

        
import os
import re
import shutil

# ---------------------------
# Contraction Replacement Rules
# ---------------------------
CONTRACTIONS = {
    "I'm": "I_am",
    "i'm": "i_am",
    "I've": "I_have",
    "i've": "i_have",
    "don't": "do_not",
    "can't": "can_not",
    "won't": "will_not",
    "didn't": "did_not",
    "shouldn't": "should_not",
    "isn't": "is_not",
    "aren't": "are_not",
    "weren't": "were_not",
    "couldn't": "could_not"
}

# ---------------------------
# Clean filename function
# ---------------------------
def clean_filename(name: str) -> str:
    original = name

    # skip DS_Store or hidden files
    if name.lower().endswith(".ds_store"):
        return None

    base, ext = os.path.splitext(name)

    # 1. Contractions
    for wrong, right in CONTRACTIONS.items():
        base = base.replace(wrong, right)

    # 1B. Replace 'swi' with 'swift' as a standalone word (case-insensitive)
    # ONLY when it's a separate word and NOT inside other words (e.g., swim)
    base = re.sub(r"\bswi\b", "swift", base, flags=re.IGNORECASE)

    # 2. Remove "feat", "feat.", "ft", "ft."
    base = re.sub(r"(feat|feat\.|ft|ft\.)", "", base, flags=re.IGNORECASE)

    # 3. Remove parentheses/brackets/braces
    base = re.sub(r"[\(\)\[\]\{\}]", "", base)

    # 4. Remove apostrophes
    base = base.replace("'", "").replace("’", "")

    # 5. Remove ALL non-English symbols
    base = re.sub(r"[^A-Za-z0-9_\s]", "", base)

    # 6. Replace spaces with underscores
    base = re.sub(r"\s+", "_", base)

    # 7. Remove multiple underscores
    base = re.sub(r"_+", "_", base)

    # 8. Remove leading/trailing underscores
    base = base.strip("_")

    # 9. Remove non-ASCII characters
    base = base.encode("ascii", "ignore").decode()

    # 10. Prevent empty filename
    if base == "":
        base = "file"

    cleaned = base + ext.lower()
    return cleaned


# ---------------------------
# Main folder processor
# ---------------------------
def clean_all_files(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            old_path = os.path.join(root, filename)

            new_name = clean_filename(filename)
            if new_name is None:
                print(f"Deleting hidden file: {old_path}")
                os.remove(old_path)
                continue

            new_path = os.path.join(root, new_name)

            if old_path != new_path:
                print(f"\nRenaming:\n{old_path}\n → {new_path}\n")
                os.rename(old_path, new_path)

    print("\n✔ DONE — All filenames cleaned.\n")


# ---------------------------
# RUN HERE — USE YOUR EXACT PATH
# ---------------------------
if __name__ == "__main__":
    folder = r"C:\Users\Prath\OneDrive\Desktop\Documents\GitHub\Data_Science_Foundation_Final_Project\Data"
    clean_all_files(folder)

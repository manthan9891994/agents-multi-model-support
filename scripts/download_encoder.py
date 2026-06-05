"""Dev helper — refresh the bundled all-MiniLM-L6-v2 encoder.

The encoder is already shipped inside the package
(classifier/ml/models/all-MiniLM-L6-v2/) so end users never need this script.
Maintainers run it when upgrading the bundled encoder to a newer revision:

    python scripts/download_encoder.py

Then review the diff and commit the refreshed folder.
"""

from pathlib import Path

SAVE_DIR = (
    Path(__file__).resolve().parent.parent
    / "classifier" / "ml" / "models" / "all-MiniLM-L6-v2"
)


def main() -> None:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Install first:  pip install sentence-transformers")
        return

    print("Downloading all-MiniLM-L6-v2 from Hugging Face...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Saving to: {SAVE_DIR}")
    model.save(str(SAVE_DIR))

    test_vec = model.encode(["adverse event safety signal"])
    print(f"Sanity check — embedding shape: {test_vec.shape}")  # (1, 384)
    print("Done. Review the diff and commit the refreshed folder.")


if __name__ == "__main__":
    main()

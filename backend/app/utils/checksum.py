import hashlib
from pathlib import Path


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def verify_sha256(file_path: Path, checksum_file: Path) -> bool:
    if not file_path.exists():
        return False

    if not checksum_file.exists():
        return False

    expected_checksum = checksum_file.read_text().strip()
    actual_checksum = calculate_sha256(file_path)

    return actual_checksum == expected_checksum
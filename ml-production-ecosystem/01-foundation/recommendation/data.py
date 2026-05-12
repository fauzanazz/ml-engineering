"""MovieLens data preparation and validation."""

from pathlib import Path
import argparse
import csv
import urllib.request
import zipfile

MOVIELENS_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
REQUIRED_RATINGS_COLUMNS = {"userId", "movieId", "rating", "timestamp"}
REQUIRED_MOVIES_COLUMNS = {"movieId", "title", "genres"}
EXPECTED_ARCHIVE_PREFIX = "ml-25m/"


class DatasetValidationError(ValueError):
    """Raised when MovieLens files are missing required shape."""


def _read_header(path: Path) -> set[str]:
    if not path.exists():
        raise DatasetValidationError(f"Missing required file: {path}")
    with path.open(newline="") as file:
        reader = csv.reader(file)
        try:
            return set(next(reader))
        except StopIteration as exc:
            raise DatasetValidationError(f"Empty CSV file: {path}") from exc


def validate_movielens_files(ratings_path: Path, movies_path: Path) -> None:
    ratings_columns = _read_header(ratings_path)
    movies_columns = _read_header(movies_path)

    missing_ratings = REQUIRED_RATINGS_COLUMNS - ratings_columns
    missing_movies = REQUIRED_MOVIES_COLUMNS - movies_columns
    if missing_ratings or missing_movies:
        raise DatasetValidationError(
            f"Invalid MovieLens files. Missing ratings columns={sorted(missing_ratings)}, "
            f"movies columns={sorted(missing_movies)}"
        )


def download_movielens_25m(data_dir: Path, url: str = MOVIELENS_25M_URL) -> Path:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ml-25m.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)
    return zip_path


def safe_extract_movielens(zip_path: Path, data_dir: Path) -> Path:
    extract_root = data_dir / "raw"
    target_dir = extract_root / "ml-25m"
    if target_dir.exists():
        return target_dir

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = member.filename
            if not name.startswith(EXPECTED_ARCHIVE_PREFIX) or ".." in Path(name).parts or name.startswith("/"):
                raise DatasetValidationError(f"Unsafe archive member: {name}")
        archive.extractall(extract_root)
    return target_dir


def prepare_movielens_25m(data_dir: Path) -> tuple[Path, Path]:
    zip_path = download_movielens_25m(data_dir)
    dataset_dir = safe_extract_movielens(zip_path, data_dir)
    ratings_path = dataset_dir / "ratings.csv"
    movies_path = dataset_dir / "movies.csv"
    validate_movielens_files(ratings_path, movies_path)
    return ratings_path, movies_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, unpack, and validate MovieLens 25M.")
    parser.add_argument("--data-dir", type=Path, default=Path("01-foundation/data"))
    args = parser.parse_args()
    ratings_path, movies_path = prepare_movielens_25m(args.data_dir)
    print(f"ratings_path={ratings_path}")
    print(f"movies_path={movies_path}")


if __name__ == "__main__":
    main()

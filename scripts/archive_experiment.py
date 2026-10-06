"""Read-only snapshot of an existing experiment into a NEW ZIP; no training."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def archive_experiment(source, output, includes=("runs", "reports", "predictions", "weights")):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir() or output == source or source in output.parents:
        raise ValueError("Source must exist and output must be OUTSIDE the source tree")
    files = []
    for name in includes:
        if Path(name).name != name:
            raise ValueError("Include names must be immediate experiment subfolders")
        folder = source / name
        if folder.is_dir():
            files += sorted(p for p in folder.rglob("*") if p.is_file())
    if not files:
        raise ValueError("No experiment files found to archive")
    if any(p.is_symlink() or source not in p.resolve().parents for p in files):
        raise ValueError("Archive refuses symlinks/paths outside the source")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
        for index, path in enumerate(files, 1):
            relative = path.relative_to(source).as_posix()
            digest = hashlib.sha256()
            stat = path.stat()
            with path.open("rb") as incoming, archive.open(relative, "w", force_zip64=True) as outgoing:
                for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                    digest.update(chunk); outgoing.write(chunk)
            if path.stat().st_size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
                raise RuntimeError("Experiment changed during archive; stop active training before backing up")
            manifest[relative] = {"sha256": digest.hexdigest(), "bytes": stat.st_size}
            print(f"Archived {index}/{len(files)}: {relative}", flush=True)
        archive.writestr("archive_manifest.json", json.dumps({"schema_version": 1, "files": manifest}, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive_experiment(args.source, args.output)
    print(f"Snapshot: {args.output}")


if __name__ == "__main__":
    main()

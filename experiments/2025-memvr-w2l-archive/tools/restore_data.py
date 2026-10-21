"""Restore or verify the archived data layout using public ZIPs and saved results."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def destination(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts:
        raise ValueError("Unsafe manifest path")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Destination escapes the data directory")
    return path


def main(manifest_override=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llava-root", type=Path, required=True)
    parser.add_argument("--downloads", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-existing", action="store_true")
    mode.add_argument("--check-inputs", action="store_true")
    mode.add_argument("--restore", action="store_true")
    parser.add_argument("--skip-results", action="store_true",
                        help="Only for read-only checking of public input sources")
    args = parser.parse_args()
    if args.skip_results and not args.check_inputs:
        parser.error("--skip-results is only valid with --check-inputs")
    archive = Path(__file__).resolve().parents[1]
    manifest = manifest_override
    if manifest is None:
        manifest = json.loads((archive / "evidence/data-recovery.json").read_text())
    data_root = args.llava_root.resolve() / "playground/data"
    files = manifest["files"]
    if args.verify_existing:
        failures = []
        for row in files:
            target = destination(data_root, row["path"])
            if not target.is_file() or blob(target.read_bytes()) != row["git_blob_sha"]:
                failures.append(row["path"])
        if failures:
            raise ValueError("Missing or different files: " + ", ".join(failures[:8]))
        print(json.dumps({"verified_files": len(files)}))
        return
    if args.downloads is None:
        parser.error("--downloads is required to check inputs or restore")
    zips, index, archive_versions = [], {}, {}
    try:
        for name, metadata in manifest["archives"].items():
            path = args.downloads / PurePosixPath(name).name
            if not path.is_file():
                path = args.downloads / name
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            archive_versions[name] = {
                "same_zip_bytes_as_server": digest.hexdigest() == metadata["sha256"]
            }
            package = zipfile.ZipFile(path)
            zips.append(package)
            for member in package.infolist():
                parts = PurePosixPath(member.filename).parts
                if member.is_dir() or "__MACOSX" in parts or parts[-1].startswith("._"):
                    continue
                index.setdefault(blob(package.read(member)), (package, member))
        plan, failures, skipped = [], [], 0
        for row in files:
            relative, expected = row["path"], row["git_blob_sha"]
            target = destination(data_root, relative)
            if expected in index:
                provider = index[expected]
            elif any(s["kind"] == "LLaVA Git baseline" for s in row["sources"]):
                source_path = manifest["baseline_data_prefix"] + relative
                data = subprocess.check_output([
                    "git", "-C", str(args.llava_root), "show",
                    manifest["baseline_commit"] + ":" + source_path,
                ], stderr=subprocess.PIPE)
                provider = data
            elif not row["sources"] and relative.startswith("eval/MME/"):
                if args.skip_results:
                    skipped += 1
                    continue
                provider = (archive / "results/MME" / relative[len("eval/MME/"):]).read_bytes()
            else:
                failures.append(relative)
                continue
            if isinstance(provider, bytes) and blob(provider) != expected:
                failures.append(relative)
                continue
            if args.restore and target.exists() and (
                not target.is_file() or blob(target.read_bytes()) != expected
            ):
                raise ValueError("Existing file differs; restore into a clean checkout: " + relative)
            plan.append((target, provider, expected))
        if failures:
            raise ValueError("ZIP contents do not match the historical data: " + ", ".join(failures[:8]))
        if args.restore:
            for target, provider, expected in plan:
                content = provider if isinstance(provider, bytes) else provider[0].read(provider[1])
                if blob(content) != expected:
                    raise ValueError("Input changed during restore")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        print(json.dumps({"mode": "restore" if args.restore else "check-inputs",
                          "matched_files": len(plan), "skipped_archived_results": skipped,
                          "archive_versions": archive_versions}, indent=2))
    finally:
        for package in zips:
            package.close()


if __name__ == "__main__":
    main()

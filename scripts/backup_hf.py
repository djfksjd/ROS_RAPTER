"""Back up committed Git history only; requires huggingface_hub."""
import subprocess
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    if git("status", "--porcelain").strip():
        raise SystemExit("Commit reviewed changes before backing up.")
    # Reject sensitive files or generated directories anywhere in Git history.
    paths = git("log", "--all", "--pretty=format:", "--name-only").decode().splitlines()
    for name in paths:
        p = Path(name)
        if (p.name == ".env" or p.name.startswith(".env.")
                or p.suffix in {".key", ".pem"}
                or {"build", "install", "log"}.intersection(p.parts)):
            raise SystemExit("Excluded path found in history; backup stopped.")
    token = None
    for line in (root / ".env").read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key == "HF_TOKEN":
            token = value.strip().strip("\"'")
    if not token:
        raise SystemExit("HF_TOKEN is not configured.")
    api = HfApi(token=token)
    repo = "dannykim123/ROS_RAPTER-backup"
    # Bundle payloads include binary Git objects; preserve existing Hub rules.
    info = api.dataset_info(repo)
    attributes = Path(hf_hub_download(repo, ".gitattributes", repo_type="dataset",
                                     revision=info.sha, token=token)).read_text()
    rule = "raptor.bundle filter=lfs diff=lfs merge=lfs -text"
    if rule not in attributes.splitlines():
        api.upload_file(path_or_fileobj=(attributes.rstrip()+"\n"+rule+"\n").encode(),
                        path_in_repo=".gitattributes", repo_id=repo, repo_type="dataset",
                        parent_commit=info.sha,
                        commit_message="Store Git backup bundle through large-file storage")
    sha = git("rev-parse", "HEAD").decode().strip()
    with tempfile.TemporaryDirectory(prefix="raptor-backup-") as tmp:
        bundle = Path(tmp) / "raptor.bundle"
        git("bundle", "create", str(bundle), "--all")
        git("bundle", "verify", str(bundle))
        api.upload_file(path_or_fileobj=bundle, path_in_repo="raptor.bundle",
                        repo_id=repo, repo_type="dataset",
                        commit_message=f"Backup ROS_RAPTER {sha}")
        info = api.dataset_info(repo, files_metadata=True)
        assert any(f.rfilename == "raptor.bundle" and f.size == bundle.stat().st_size
                   for f in info.siblings), "Remote backup size verification failed"
    print(f"HF backup verified: {repo} / Git HEAD {sha}")


if __name__ == "__main__":
    main()

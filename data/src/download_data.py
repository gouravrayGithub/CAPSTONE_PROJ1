from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="phamquiluan/RCAEval",
    repo_type="dataset",
    allow_patterns="re2ob_*",
    local_dir="data"
)

print("RE2-OB dataset downloaded.")
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="BAAI/bge-m3",  
    local_dir="bge-m3",
    local_dir_use_symlinks=False,
    resume_download=True,      # important: resume instead of starting from scratch
    max_workers=4              # fewer parallel connections can help on weak networks
)

print("Downloaded to:", local_dir)

# https://huggingface.co/Alibaba-NLP/gte-multilingual-basefsdf
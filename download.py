from huggingface_hub import snapshot_download

def main():
    snapshot_download(
        repo_id="hunyuanvideo-community/HunyuanVideo",
        local_dir="Path/to/where/you/want/to/save/model",
        local_dir_use_symlinks=False
    )

if __name__ == "__main__":
    main()
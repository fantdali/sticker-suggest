from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(default=None)
    sticker_dir: Path = field(default=None)
    embeddings_dir: Path = field(default=None)
    # CLIP
    clip_model: str = "ViT-L-14"
    clip_pretrained: str = "openai"
    clip_device: str = "mps"
    embedding_dim: int = 768
    batch_size: int = 64

    # Deduplication
    hash_size: int = 8
    hash_func: str = "phash"

    # OCR
    ocr_languages: list[str] = field(default_factory=lambda: ["en"])
    ocr_gpu: bool = True
    ocr_confidence_threshold: float = 0.3
    ocr_max_text_len: int = 100

    # Retrieval
    top_k: int = 10
    ocr_alpha: float = 0.01

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.sticker_dir is None:
            self.sticker_dir = self.data_dir / "sticker_suggest"
        if self.embeddings_dir is None:
            self.embeddings_dir = self.data_dir / "embeddings"

    def ensure_dirs(self):
        for d in [self.data_dir, self.embeddings_dir]:
            d.mkdir(parents=True, exist_ok=True)

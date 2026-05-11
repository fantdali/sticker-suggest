from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(default=None)
    sticker_dir: Path = field(default=None)
    embeddings_dir: Path = field(default=None)
    output_dir: Path = field(default=None)
    generated_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
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
    generation_min_score: float = 4.0

    # Generation
    generation_model: str = "stabilityai/sdxl-turbo"
    generation_device: str = "auto"
    generation_local_files_only: bool = True
    generation_variant: str = "fp16"
    generation_width: int = 512
    generation_height: int = 512
    generation_steps: int = 4
    generation_guidance_scale: float = 0.0
    sam_checkpoint: Path = field(default=None)
    sam_model_type: str = "vit_l"
    sam_device: str = "cpu"
    outline_size: int = 3

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.sticker_dir is None:
            self.sticker_dir = self.data_dir / "sticker_suggest"
        if self.embeddings_dir is None:
            self.embeddings_dir = self.data_dir / "embeddings"
        if self.output_dir is None:
            self.output_dir = self.project_root / "outputs"
        if self.generated_dir is None:
            self.generated_dir = self.output_dir / "generated"
        if self.models_dir is None:
            self.models_dir = self.project_root / "models"
        if self.sam_checkpoint is None:
            self.sam_checkpoint = self.models_dir / "sam_vit_l_0b3195.pth"

    def ensure_dirs(self):
        for d in [self.data_dir, self.embeddings_dir, self.output_dir, self.generated_dir]:
            d.mkdir(parents=True, exist_ok=True)

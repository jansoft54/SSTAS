from dataclasses import dataclass




@dataclass
class TrainerConfig:
    dataset: str = "50salads"
    train_split: str = "train.split1.bundle"
    test_split: str = "test.split1.bundle"

    default_path: str = "./data/data/"
    output_name: str = "output"

    batch_size: int = 1
    knowns: int = 0
    unknowns: int = 0
    K: int = 0

    learning_rate: float = 2e-4
    epsilon: float = 1e-8
    num_epochs: int = 50
    mask_ratio: float = 0.85
    block_size: int = 64
    min_span: int = 32
    max_span: int = 64

    pim_loss_weight: float = 1.0
    recon_loss_weight: float = 0
    smooth_loss_weight: float = 1
    cosine_recon_weight: float = 1.0
    l1_recon_weight: float = 1.0

    info_nce_weight: float = 0
    info_nce_temp: float = 0
    info_start: float = 0

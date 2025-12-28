from dataclasses import dataclass


@dataclass
class TrainerConfig:
    dataset: str = "50salads"
    train_split: str = "train.split1.bundle"
    test_split: str = "test.split1.bundle"

    default_path: str = "./data/data/"
    output_name: str = "output"

    batch_size: int = 1
    knowns: int = 14
    unknowns: int = 5
    K: int = 30

    learning_rate: float = 2e-4
    epsilon: float = 1e-8
    weight_decay:float = 0.02
    
    num_epochs: int = 50
    mask_ratio: float = 0.8
    block_size: int = 128
    
    min_span: int = 32
    max_span: int = 64

    pim_loss_weight: float = 1
    recon_loss_weight: float = 1
    smooth_loss_weight: float = 1
    cosine_recon_weight: float = 1
    l1_recon_weight: float = 1

    info_nce_weight: float = 0
    info_nce_temp: float = 0
    info_start: float = 0

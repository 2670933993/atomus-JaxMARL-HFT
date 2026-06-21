"""
LOBSTER-related constants and Mamba model configuration dataclasses.

LOBSTER column names and token constants are now centralized in
gymnax_exchange/config/constants.py and re-exported here for
backward compatibility.

The Mamba dataclasses live here because they are specific to
LOBSTER-based model training/inference tasks.
"""

from dataclasses import dataclass
import numpy as np

# Re-export LOBSTER token constants from centralized location
from gymnax_exchange.config.constants import (           # noqa: F401
    TIME_COL,
    EVENT_TYPE_COL,
    ORDER_ID_COL,
    SIZE_COL,
    PRICE_COL,
    DIRECTION_COL,
    MESSAGE_TOKEN_DTYPE_MAP,
    MESSAGE_TOKEN_TYPES,
)


def get_orderbook_token_types(levels: int) -> list[str]:
    """Generate token type names for orderbook levels."""
    return np.array([
        [f"<ask_price_{i}>", f"<ask_size_{i}>", f"<bid_price_{i}>", f"<bid_size_{i}>"]
        for i in range(1, levels + 1)]
    ).flatten().tolist()


# ──────────────────────────────────────────────
# Mamba / Tokenizer Configuration
# ──────────────────────────────────────────────

@dataclass
class MambaTrainArgs:
    train_data_dir: str = "./data/raw/"
    eval_data_dir: str = "./data/test/"
    file_filter_train: str = ""
    file_filter_eval: str = ""
    save_path: str = "./models/mamba2"
    nmsgs: int = 50
    only_use_message_orderbook_matches: bool = True

    tokenizer_file: str = "tokenizers/lob_tok_with_time_diff.json"

    wandb_online: bool = True
    wandb_project: str = "lobgen"
    wandb_entity: str = "gereon-franken-oxford"


@dataclass
class MambaInferenceArgs:
    model_path: str
    tokenizer_path: str = "tokenizers/lob_tok_messages.json"
    is_sharded: bool = False
    test_dir: str = "data/GOOG/test/"
    test_filter: str = "2018-12-31"
    genlen: int = 100
    iterations: int = 10
    temperature: float = 1.0
    topk: int = 50
    topp: float = 1.0
    minp: float = 0.0
    repetition_penalty: float = 1.0
    batch: int = 1


@dataclass
class MambaBenchmarkingArgs(MambaInferenceArgs):
    data_dir: str = "data/GOOG/2018/"
    data_time_stamp: str = "2018-12-31"
    save_path: str = "gen_data/"


@dataclass
class TokenizerTrainArgs:
    data_dir: str = "./data/raw/"
    file_filter: str = "*.csv"
    save_path: str = "./tokenizers/lob_tok.json"
    vocab_size: int = 10_000

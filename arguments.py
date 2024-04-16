from typing import List, Optional, Tuple, Union

from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class CustomTrainingArguments(TrainingArguments):
    # experiment setups
    reward_domain: str = field(
        default="normal", metadata={"help": "the domain for reward model training."}
    )
    # tokenizer params
    padding_side: str = field(
        default="right",
        metadata={"help": "the direction for tokenizer to add padding tokens."},
    )

    truncation_side: str = field(
        default="left",
        metadata={"help": "the direction for tokenizer to add padding tokens."},
    )

    # model params
    model_type: str = field(
        default="llama",
        metadata={
            "help": "the base model type for reward model, selected from [llama, bert, pythia]."
        },
    )

    pooling_type: str = field(
        default="average",
        metadata={
            "help": "the pooling method for reward model, selected from [average, max, last]."
        },
    )

    model_name_or_path: str = field(
        default="llama-7b-hf", metadata={"help": "the path to load pretrained model."}
    )

    tokenizer_path: str = field(
        default="llama-7b-hf",
        metadata={"help": "the path to load pretrained tokenizer."},
    )

    # data params
    max_response_num: int = field(
        default=1, metadata={"help": "the maximum response number of each data item"}
    )

    data_dir: str = field(
        default="path/to/cleaned_data", metadata={"help": "the directory to load data."}
    )

    data_path: str = field(
        default="yahma/alpaca-cleaned", metadata={"help": "the path to load data."}
    )

    train_data_path: List[str] = field(
        default_factory=lambda: ["./data/"],
        metadata={"help": "train datasets paths."},
    )

    eval_data_path: List[str] = field(
        default_factory=lambda: ["./data/"],
        metadata={"help": "evaluation datasets paths."},
    )

    data_prefix: str = field(
        default="",
        metadata={"help": "the prefix to load train and test data."},
    )

    # training hyperparams
    eval_at_start: bool = field(
        default=False, metadata={"help": "whether make eval at start."}
    )

    debug_mode: bool = field(
        default=False, metadata={"help": "whether use the debug mode."}
    )

    cache_dir: Optional[str] = field(default=None)

    optim: str = field(default="adamw_torch")

    lm_loss_coeff: float = field(
        default=0.0, metadata={"help": "the coefficient for language modeling loss."}
    )

    # data composition
    pair_data_augmentation: Optional[bool] = field(
        default=False, metadata={"help": "method of hybrid datasets"}
    )

    shuffle: Optional[bool] = field(default=False, metadata={"help": "shuffle hybrid"})

    resampling_size: Optional[int] = field(
        default=1000, metadata={"help": "shuffle hybrid"}
    )
    resampling: Optional[bool] = field(
        default=False, metadata={"help": "shuffle hybrid"}
    )

    # multi objective optimization
    run_more: Optional[bool] = field(default=False, metadata={"help": "if start more"})

    more: Optional[bool] = field(
        default=False, metadata={"help": "multi objective mode"}
    )
    normalize: Optional[str] = field(
        default="none", metadata={"help": "multi objective mode"}
    )
    reweight: Optional[bool] = field(
        default=True, metadata={"help": "multi objective mode"}
    )
    alpha: Optional[float] = field(default=1, metadata={"help": "lambda momentum"})
    task_num: Optional[int] = field(
        default=2, metadata={"help": "number of objectives"}
    )

    calibration: Optional[bool] = field(
        default=False, metadata={"help": "multi objective mode"}
    )

    # calibration_bins: Optional[list] = field(default=[5,10,20], metadata={"help": "multi objective mode"})

    # efficient
    active_module_name: Optional[str] = field(
        default="none", metadata={"help": "multi objective mode"}
    )

    contrast_loss_coeff: float = field(
        default=0.0, metadata={"help": "the coefficient for contrastive learning loss."}
    )

    lm_score_thresh: float = field(
        default=0.85,
        metadata={"help": "the threshold to select response for language modeling"},
    )

    max_length: int = field(
        default=256, metadata={"help": "the max sentence sequence length."}
    )

    batch_size: int = field(
        default=256, metadata={"help": "the overall training batch size"}
    )

    micro_batch_size: int = field(
        default=32,
        metadata={
            "help": "the batch size on each device, equavilent to `per_gpu_train_batch_size`"
        },
    )

    valid_data_size: int = field(
        default=0, metadata={"help": "the data size for validation data"}
    )

    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "either training checkpoint or final adapter"}
    )

    # logging
    logging_dir: str = field(
        default="./tb_logs", metadata={"help": "the directory to save logs."}
    )

    logging_strategy: str = field(
        default="steps", metadata={"help": "the strategy to save logs."}
    )

    logging_steps: int = field(
        default=100, metadata={"help": "the steps to print logs."}
    )
    report_to: str = field(
        default="tensorboard", metadata={"help": "the directory to save logs."}
    )

    sep_token: Optional[str] = field(
        default="<sep>",
        metadata={
            "help": "the token that can use to seperate the query and answer in text"
        },
    )

    only_predict_answer: Optional[bool] = field(
        default=True, metadata={"help": "Only predict the answer."}
    )

    pad_labels_with_ignore: Optional[bool] = field(
        default=False, metadata={"help": "Whether use ignore token to pad labels."}
    )

    ignore_token_id: Optional[int] = field(
        default=-100, metadata={"help": "token id used to inplace query ids."}
    )

    beta: Optional[float] = field(
        default=0.1, metadata={"help": "token id used to inplace query ids."}
    )

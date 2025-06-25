import argparse
import ast

from transformers import AutoTokenizer, PreTrainedTokenizer

from curriculum_training.constants import NWR_TRAINING_V3
from curriculum_training.curriculum_utils import DifficultyLevels
from curriculum_training.curriculum_training import curriculum_trianing_main
from utils import hook_stdout

from transformers import Qwen2ForCausalLM
# from custome_model import CustomQwen2Config, Qwen2ForCausalLM


def parse_args():
    """
    Parse command line arguments.
    Example usage:
    python curriculum_custome.py
    --limit_news 5000
    --dataset_name nwr_training.json
    --stages_list [[1], [0,1,2,3,4]]
    --training
    """
    parser = argparse.ArgumentParser(description="Curriculum Training Script")
    parser.add_argument("--limit_news", "-n", type=int,
                        help="Limit the size of training data", default=None)
    parser.add_argument("--dataset_name", "-d", type=str,
                        help="Dataset name", default=None)
    parser.add_argument("--stages_list", type=str,
                        help="AsStages list of a list of lists", default=None)
    parser.add_argument("--training", "-t", action="store_true",
                        help="Training mode", default=None)
    parser.add_argument("--no_training", "-nt", action="store_false",
                        help="Training mode", default=None)
    parser.add_argument("--model_path", "-m", type=str,
                        help="Model path", default=None)
    return parser.parse_args()


def create_model(config, model_path) -> None:
    """
    Create model and tokenizer to specified path.
    """
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(model_path, safe_serialization=False)
    # create tokenizer(use qwen2's tokenizer)
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-0.5B-Instruct"
    )
    # assert isinstance(tokenizer, PreTrainedTokenizer)
    tokenizer.bos_token_id = config.bos_token_id
    tokenizer.eos_token_id = config.eos_token_id
    tokenizer.pad_token_id = config.pad_token_id or tokenizer.pad_token_id
    tokenizer.save_pretrained(model_path)


if __name__ == "__main__":
    hook_stdout()

    stages_list = [
        [
            DifficultyLevels.DIRECT_SUMMARY,
        ],
        [
            DifficultyLevels.ESSENTIAL_ASPECTS,
            DifficultyLevels.TRIPLES,
            DifficultyLevels.SUMMARY,
            DifficultyLevels.DIRECT_SUMMARY,
        ],
        [
            DifficultyLevels.TO_ZHT,
            DifficultyLevels.ESSENTIAL_ASPECTS,
            DifficultyLevels.TRIPLES,
            DifficultyLevels.SUMMARY,
            DifficultyLevels.DIRECT_SUMMARY,
        ]
    ]

    limit_news = None
    dataset_name = NWR_TRAINING_V3  # v3
    to_train = True
    model_path = "CustomQwen2Model_pretrained"

    args = parse_args()

    if args.limit_news is not None:
        limit_news = args.limit_news

    if args.dataset_name is not None:
        dataset_name = args.dataset_name

    if args.training is not None:
        to_train = args.training
    if args.no_training is not None:
        to_train = args.no_training

    if args.stages_list is not None:
        stages_list_int = ast.literal_eval(args.stages_list)
        stages_list = []
        for stage in stages_list_int:
            stages_list.append([DifficultyLevels(i) for i in stage])

    if args.model_path is not None:
        model_path = args.model_path

    # create_model(
    #     # CustomQwen2Config(
    #     Qwen2Config(
    #         hidden_size=896,
    #         num_attention_heads=14,
    #         num_hidden_layers=24,
    #     ),
    #     model_path
    # )

    curriculum_trianing_main(
        model_path=model_path,
        dataset_name=dataset_name,
        limit_news=limit_news,
        stages_list=stages_list,
        to_train=to_train,
    )

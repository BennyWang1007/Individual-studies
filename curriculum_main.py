import argparse
import ast

from curriculum_training.constants import (
    MODEL_BASE,
    NWR_TRAINING_FILE, NWR_TRAINING_V2, NWR_TRAINING_V3, NWR_TRAINING_V4,
)
from curriculum_training.curriculum_utils import DifficultyLevels
from curriculum_training.curriculum_training import curriculum_trianing_main
from utils import hook_stdout

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
        ],
    ]

    limit_news = None  # None means no limit
    dataset_name = NWR_TRAINING_FILE  # v1
    dataset_name = NWR_TRAINING_V2  # v2
    dataset_name = NWR_TRAINING_V3  # v3
    dataset_name = NWR_TRAINING_V4  # v4

    to_train = True

    # parse the command line arguments, e.g.:
    # --limit_news 5000 --dataset_name nwr_training.json
    # --stages_list [[1], [0,1,2,3,4]]
    parser = argparse.ArgumentParser(description="Curriculum Training Script")
    parser.add_argument("--limit_news", "-n", type=int,
                        help="Limit the size of training data", default=None)
    parser.add_argument("--dataset_name", "-d", type=str,
                        help="Dataset name", default=None)
    parser.add_argument("--stages_list", type=str,
                        help="AsStages list of a list of lists", default=None)

    parser.add_argument("--train", "-t", action="store_true",
                        help="Enable training mode")
    parser.add_argument("--not_train", "-nt", action="store_true",
                        help="Disable training mode")

    args = parser.parse_args()

    if args.limit_news is not None:
        limit_news = args.limit_news

    if args.dataset_name is not None:
        dataset_name = args.dataset_name

    if args.train:
        to_train = True
        print("Training mode enabled")
    elif args.not_train:
        to_train = False
        print("Training mode disabled")

    if args.stages_list is not None:
        stages_list_int = ast.literal_eval(args.stages_list)
        stages_list = []
        for stage in stages_list_int:
            stages_list.append([DifficultyLevels(i) for i in stage])

    # print(f"Limit news: {limit_news}")
    # print(f"Dataset name: {dataset_name}")
    # print(f"Stages list: {stages_list}")
    # print(f"Training: {to_train}")
    # print(f"Model base: {MODEL_BASE}")

    curriculum_trianing_main(
        model_path=MODEL_BASE,
        dataset_name=dataset_name,
        limit_news=limit_news,
        stages_list=stages_list,
        to_train=to_train,
    )

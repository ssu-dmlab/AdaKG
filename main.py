import sys
import argparse
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.config.configurator import Config
from recbole.data import create_dataset, data_preparation
from model.AdaKG import AdaKG
from trainer.adakg_trainer import AdaKGTrainer


parser = argparse.ArgumentParser(description="AdaKG")
parser.add_argument(
    "--dataset",
    type=str,
    default="lastfm",
    choices=["ml-1m", "lastfm", "Amazon-book", "book-crossing"]
)

args = parser.parse_args()
sys.argv = sys.argv[:1]

config = Config(
    model=AdaKG,
    dataset=args.dataset,
    config_file_list=[f"./yaml/{args.dataset}_AdaKG.yaml"]
)

init_seed(config["seed"], config["reproducibility"])
init_logger(config)
logger = getLogger()
data = create_dataset(config)
logger.info(data)

train_data, valid_data, test_data = data_preparation(config, data)

model = AdaKG(
    config=config,
    dataset=train_data._dataset
).to(config["device"])

logger.info(model)
trainer = AdaKGTrainer(config, model)
best_valid_score, best_valid_result = trainer.fit(
    train_data,
    valid_data
)

test_at_best = trainer.evaluate(
    test_data,
    load_best_model=True
)

logger.info(f"[BEST-VALID] : {best_valid_result} " f"(score={best_valid_score:.6f})")
logger.info(f"[TEST] : {test_at_best}")
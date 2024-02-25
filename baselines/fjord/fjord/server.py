"""Global evaluation function."""

from typing import Any, Dict, Optional, Tuple

import flwr as fl
import torch
from torch.utils.data import DataLoader

from .dataset import load_data
from .datasets import get_num_clients
from .datasets.utils import prepare_dataset
from .models import get_net, test
from .utils.logger import Logger
from .utils.utils import save_model, set_parameters


def get_eval_fn(
    args: Any, model_path: str, testloader: DataLoader, device: torch.device
):
    """Get evaluation function.

    :param args: Arguments
    :param model_path: Path to save the model
    :param testloader: Test data loader
    :param device: Device to be used
    :return: Evaluation function
    """

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],  # pylint: disable=unused-argument
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        if server_round and (server_round % args.evaluate_every == 0):
            net = get_net(args.model, args.p_s, device)
            set_parameters(net, parameters)
            fed_dir, globaldata_dir = prepare_dataset(args.dataset, args.data_path, lda_alpha=args.lda,
                                                      number_of_clients=get_num_clients(args.dataset))
            path = fed_dir
            if args.eval_every_client:
                for cid in range(args.num_clients):
                    _, testloader = load_data(
                        args.dataset, path, cid=cid, seed=args.manual_seed, train_bs=args.batch_size
                    )
                    losses, accuracies = test(net, testloader, args.p_s)
                    avg_loss = sum(losses) / len(losses)
                    for p, loss, accuracy in zip(args.p_s, losses, accuracies):
                        Logger.get().info(
                            f"Client-side evaluation (global round={server_round}, cid={cid})"
                            f" {p=}: {loss=} / {accuracy=}"
                        )
            else:
                # Update model with the latest parameters
                losses, accuracies = test(net, testloader, args.p_s)
                avg_loss = sum(losses) / len(losses)
                for p, loss, accuracy in zip(args.p_s, losses, accuracies):
                    Logger.get().info(
                        f"Server-side evaluation (global round={server_round})"
                        f" {p=}: {loss=} / {accuracy=}"
                    )
            save_model(net, model_path)

            return avg_loss, {
                f"Accuracy[{p}]": acc for p, acc in zip(args.p_s, accuracies)
            }

        Logger.get().debug(f"Evaluation skipped for global round={server_round}.")
        return float("inf"), {"accuracy": "None"}

    return evaluate

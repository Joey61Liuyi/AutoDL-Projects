#####################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.01 #
#####################################################
import sys, time, torch, random, argparse
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path

from xautodl.datasets import get_datasets
from xautodl.config_utils import load_config, obtain_basic_args as obtain_args
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
)
from xautodl.procedures import get_optim_scheduler, get_procedures
from xautodl.models import obtain_model
from xautodl.nas_infer_model import obtain_nas_infer_model
from xautodl.utils import get_model_infos
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
import numpy as np
from torch.utils.data import Dataset
import copy

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key].float(), len(w))
    return w_avg



class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach(), torch.tensor(label)


def main(args):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.set_num_threads(args.workers)

    prepare_seed(args.rand_seed)
    logger = prepare_logger(args)

    train_data, valid_data, xshape, class_num = get_datasets(
        args.dataset, args.data_path, args.cutout_length
    )

    valid_use = False
    user_data = np.load('../../exps/NAS-Bench-201-algos/Use_valid_{}_{}_non_iid_setting.npy'.format(valid_use, args.dataset), allow_pickle=True).item()
    train_loader_list = {}
    valid_loader_list = {}
    for user in user_data:
        train_loader_list[user] = torch.utils.data.DataLoader(
                                    DatasetSplit(train_data, user_data[user]['train']+user_data[user]['test']),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.workers,
                                    pin_memory=True,
                                )
        valid_loader_list[user] = torch.utils.data.DataLoader(
                                    DatasetSplit(valid_data, user_data[user]['valid']),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.workers,
                                    pin_memory=True,
                                )

    # train_loader = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True,
    # )
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_data,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.workers,
    #     pin_memory=True,
    # )

    # get configures
    model_config = load_config(args.model_config, {"class_num": class_num}, logger)
    optim_config = load_config(args.optim_config, {"class_num": class_num}, logger)

    if args.model_source == "normal":
        base_model = obtain_model(model_config)
    elif args.model_source == "nas":
        base_model = obtain_nas_infer_model(model_config, args.extra_model_path)
    elif args.model_source == "autodl-searched":
        base_model_list = {}
        for user in user_data:
            base_model_list[user] = obtain_model(model_config, user)
        # base_model = obtain_model(model_config, args.extra_model_path)
    else:
        raise ValueError("invalid model-source : {:}".format(args.model_source))

    optimizer_list = {}
    scheduler_list = {}
    criterion_list = {}
    state_dict_list = {}
    for user in user_data:
        flop, param = get_model_infos(base_model_list[user], xshape)
        logger.log("model ====>>>>:\n{:}".format(base_model_list[user]))
        logger.log("model information : {:}".format(base_model_list[user].get_message()))
        logger.log("-" * 50)
        logger.log(
            "Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
                param, flop, flop / 1e3
            )
        )
        logger.log("-" * 50)
        optimizer_list[user], scheduler_list[user], criterion_list[user] = get_optim_scheduler(base_model_list[user].parameters(), optim_config)

        logger.log("User{}, train_data : {:}".format(user, train_data[user]))
        logger.log("User{}, valid_data : {:}".format(user, valid_data[user]))
        # optimizer, scheduler, criterion = get_optim_scheduler(
        #     base_model.parameters(), optim_config
        # )
        logger.log("User{}, optimizer  : {:}".format(user, optimizer_list[user]))
        logger.log("User{}, scheduler  : {:}".format(user, scheduler_list[user]))
        logger.log("User{}, criterion  : {:}".format(user, criterion_list[user]))
        # base_model_list[user], criterion_list[user] = torch.nn.DataParallel(base_model[user]).cuda(), criterion_list[user].cuda()
        criterion_list[user] = criterion_list[user].cuda()
        base_model_list[user] = base_model_list[user].cuda()
        state_dict_list[user] = base_model_list[user].state_dict()

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )




    # if last_info.exists():  # automatically resume from previous checkpoint
    #     logger.log(
    #         "=> loading checkpoint of the last-info '{:}' start".format(last_info)
    #     )
    #     last_infox = torch.load(last_info)
    #     start_epoch = last_infox["epoch"] + 1
    #     last_checkpoint_path = last_infox["last_checkpoint"]
    #     if not last_checkpoint_path.exists():
    #         logger.log(
    #             "Does not find {:}, try another path".format(last_checkpoint_path)
    #         )
    #         last_checkpoint_path = (
    #             last_info.parent
    #             / last_checkpoint_path.parent.name
    #             / last_checkpoint_path.name
    #         )
    #     checkpoint = torch.load(last_checkpoint_path)
    #     base_model.load_state_dict(checkpoint["base-model"])
    #     scheduler.load_state_dict(checkpoint["scheduler"])
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #     valid_accuracies = checkpoint["valid_accuracies"]
    #     max_bytes = checkpoint["max_bytes"]
    #     logger.log(
    #         "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
    #             last_info, start_epoch
    #         )
    #     )
    # elif args.resume is not None:
    #     assert Path(args.resume).exists(), "Can not find the resume file : {:}".format(
    #         args.resume
    #     )
    #     checkpoint = torch.load(args.resume)
    #     start_epoch = checkpoint["epoch"] + 1
    #     base_model.load_state_dict(checkpoint["base-model"])
    #     scheduler.load_state_dict(checkpoint["scheduler"])
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #     valid_accuracies = checkpoint["valid_accuracies"]
    #     max_bytes = checkpoint["max_bytes"]
    #     logger.log(
    #         "=> loading checkpoint from '{:}' start with {:}-th epoch.".format(
    #             args.resume, start_epoch
    #         )
    #     )
    # elif args.init_model is not None:
    #     assert Path(
    #         args.init_model
    #     ).exists(), "Can not find the initialization file : {:}".format(args.init_model)
    #     checkpoint = torch.load(args.init_model)
    #     base_model.load_state_dict(checkpoint["base-model"])
    #     start_epoch, valid_accuracies, max_bytes = 0, {"best": -1}, {}
    #     logger.log("=> initialize the model from {:}".format(args.init_model))
    # else:
    #     logger.log("=> do not find the last-info file : {:}".format(last_info))
    #     start_epoch, valid_accuracies, max_bytes = 0, {"best": -1}, {}

    start_epoch, valid_accuracies, max_bytes = 0, {"best": -1}, {}
    train_func, valid_func = get_procedures(args.procedure)

    total_epoch = optim_config.epochs + optim_config.warmup
    local_epoch = 3
    # Main Training and Evaluation Loop
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(start_epoch, total_epoch):

        global_model = average_weights(list(state_dict_list.values()))
        for user in scheduler_list:
            base_model_list[user].load_state_dict(global_model)
            scheduler_list[user].update(epoch, 0.0)
        need_time = "Time Left: {:}".format(
            convert_secs2time(epoch_time.avg * (total_epoch - epoch), True)
        )
        epoch_str = "epoch={:03d}/{:03d}".format(epoch, total_epoch)
        LRs = scheduler_list[0].get_lr()
        find_best = False
        # set-up drop-out ratio
        # if hasattr(base_model, "update_drop_path"):
        #     base_model.update_drop_path(
        #         model_config.drop_path_prob * epoch / total_epoch
        #     )
        logger.log(
            "\n***{:s}*** start {:s} {:s}, LR=[{:.12f} ~ {:.12f}], scheduler={:}".format(
                time_string(), epoch_str, need_time, min(LRs), max(LRs), scheduler_list[0]
            )
        )

        # train for one epoch

        test_accuracy_list = []

        for user in train_loader_list:
            train_loss, train_acc1, train_acc5, state_dict_list[user] = train_func(
                train_loader_list[user],
                base_model_list[user],
                criterion_list[user],
                scheduler_list[user],
                optimizer_list[user],
                optim_config,
                epoch_str,
                args.print_freq,
                logger,
                local_epoch
            )
            # log the results
            logger.log(
                "User {} ***{:s}*** TRAIN [{:}] loss = {:.6f}, accuracy-1 = {:.2f}, accuracy-5 = {:.2f}".format(
                    user, time_string(), epoch_str, train_loss, train_acc1, train_acc5
                )
            )

            # evaluate the performance
            if (epoch % 1 == 0) or (epoch + 1 == total_epoch):
                logger.log("-" * 150)
                valid_loss, valid_acc1, valid_acc5 = valid_func(
                    valid_loader_list[user],
                    base_model_list[user],
                    criterion_list[user],
                    optim_config,
                    epoch_str,
                    args.print_freq_eval,
                    logger,
                )
            valid_accuracies[epoch] = valid_acc1
            logger.log(
                "Important: User {}: ***{:s}*** VALID [{:}] loss = {:.6f}, accuracy@1 = {:.2f}, accuracy@5 = {:.2f} | Best-Valid-Acc@1={:.2f}, Error@1={:.2f}".format(
                    user,
                    time_string(),
                    epoch_str,
                    valid_loss,
                    valid_acc1,
                    valid_acc5,
                    valid_accuracies["best"],
                    100 - valid_accuracies["best"],
                )
            )
            if valid_acc1 > valid_accuracies["best"]:
                valid_accuracies["best"] = valid_acc1
                find_best = True
                logger.log(
                    "Currently, the best validation accuracy found at {:03d}-epoch :: acc@1={:.2f}, acc@5={:.2f}, error@1={:.2f}, error@5={:.2f}, save into {:}.".format(
                        epoch,
                        valid_acc1,
                        valid_acc5,
                        100 - valid_acc1,
                        100 - valid_acc5,
                        model_best_path,
                    )
                )
            test_accuracy_list.append(valid_acc1)
            info_dict = {
                         "{}user_train_loss".format(user): train_loss,
                         "{}user_train_top1".format(user): train_acc1,
                         "{}user_train_top5".format(user): train_acc5,
                         "{}user_valid_loss".format(user): valid_loss,
                         "{}user_valid_top1".format(user): valid_acc1,
                         "{}user_valid_top5".format(user): valid_acc5,
                         "epoch": epoch
                         }
            wandb.log(info_dict)
        info_dict = {
                     "average_valid_acc": np.average(test_accuracy_list),
                     "epoch": epoch
                     }
        wandb.log(info_dict)

            # num_bytes = (
            #     torch.cuda.max_memory_cached(next(network.parameters()).device) * 1.0
            # )
            # logger.log(
            #     "[GPU-Memory-Usage on {:} is {:} bytes, {:.2f} KB, {:.2f} MB, {:.2f} GB.]".format(
            #         next(network.parameters()).device,
            #         int(num_bytes),
            #         num_bytes / 1e3,
            #         num_bytes / 1e6,
            #         num_bytes / 1e9,
            #     )
            # )
            # max_bytes[epoch] = num_bytes
        if epoch % 10 == 0:
            torch.cuda.empty_cache()

        # save checkpoint
        # save_path = save_checkpoint(
        #     {
        #         "epoch": epoch,
        #         "args": deepcopy(args),
        #         "max_bytes": deepcopy(max_bytes),
        #         "FLOP": flop,
        #         "PARAM": param,
        #         "valid_accuracies": deepcopy(valid_accuracies),
        #         "model-config": model_config._asdict(),
        #         "optim-config": optim_config._asdict(),
        #         "base-model": base_model.state_dict(),
        #         "scheduler": scheduler.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #     },
        #     model_base_path,
        #     logger,
        # )
        # if find_best:
        #     copy_checkpoint(model_base_path, model_best_path, logger)
        # last_info = save_checkpoint(
        #     {
        #         "epoch": epoch,
        #         "args": deepcopy(args),
        #         "last_checkpoint": save_path,
        #     },
        #     logger.path("info"),
        #     logger,
        # )

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    logger.log("\n" + "-" * 200)
    # logger.log(
    #     "Finish training/validation in {:} with Max-GPU-Memory of {:.2f} MB, and save final checkpoint into {:}".format(
    #         convert_secs2time(epoch_time.sum, True),
    #         max(v for k, v in max_bytes.items()) / 1e6,
    #         logger.path("info"),
    #     )
    # )
    logger.log("-" * 200 + "\n")
    logger.close()

class Config():
    def __init__(self):
        self.dataset = 'cifar10'
        self.batch = 96
        self.datapath = '../../../data/{}'
        self.model_source = 'autodl-searched'
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            base = 'CIFAR'
        self.model_config = './NAS-{}-none.config'.format(base)
        self.optim_config = './NAS-{}.config'.format(base)
        self.extra_model_path = None
        self.procedure = 'basic'
        self.save_dir = './output/nas-infer/{}-BS{}-gdas-searched'.format(self.dataset, self.batch)
        self.cut_out_length = 16
        self.workers = 4
        self.seed = -1
        self.print_freq = 500
        self.print_freq_eval = 1000



if __name__ == "__main__":

    import wandb
    wandb.init(project="Federated_NAS_inference", name='cifar10_FedNAS')
    config = Config()
    parser = argparse.ArgumentParser(
        description="Train a classification model on typical image classification datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--resume", type=str, help="Resume path.")
    parser.add_argument("--init_model", type=str, help="The initialization model path.")
    parser.add_argument(
        "--model_config", type=str, default=config.model_config, help="The path to the model configuration"
    )
    parser.add_argument(
        "--optim_config", type=str, default=config.optim_config, help="The path to the optimizer configuration"
    )
    parser.add_argument("--procedure", type=str, default=config.procedure, help="The procedure basic prefix.")
    parser.add_argument(
        "--model_source",
        type=str,
        default=config.model_source,
        help="The source of model defination.",
    )
    parser.add_argument(
        "--extra_model_path",
        type=str,
        default=config.extra_model_path,
        help="The extra model ckp file (help to indicate the searched architecture).",
    )
    parser.add_argument("--dataset", type=str, default= config.dataset,help="The dataset name.")
    parser.add_argument("--data_path", type=str, default=config.datapath,  help="The dataset name.")
    parser.add_argument(
        "--cutout_length", type=int, default=config.cut_out_length, help="The cutout length, negative means not use."
    )
    # Printing
    parser.add_argument(
        "--print_freq", type=int, default=config.print_freq, help="print frequency (default: 200)"
    )
    parser.add_argument(
        "--print_freq_eval",
        type=int,
        default=config.print_freq_eval,
        help="print frequency (default: 200)",
    )
    # Checkpoints
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=1,
        help="evaluation frequency (default: 200)",
    )
    parser.add_argument(
        "--save_dir", type=str, default=config.save_dir, help="Folder to save checkpoints and log."
    )
    # Acceleration
    parser.add_argument(
        "--workers",
        type=int,
        default=config.workers,
        help="number of data loading workers (default: 8)",
    )
    # Random Seed
    parser.add_argument("--rand_seed", type=int, default=config.seed, help="manual seed")
    # Optimization options
    parser.add_argument(
        "--batch_size", type=int, default=config.batch, help="Batch size for training."
    )
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_dir is not None, "save-path argument can not be None"
    wandb.config.update(args)
    main(args)

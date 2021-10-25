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
from xautodl.nas_infer_model.DXYs.genotypes import Networks
import numpy as np
from torch.utils.data import Dataset
from Models import create_cnn_model
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


def partial_average_weights(base_model_list):
    """
    Returns the average of the weights.
    """
    w = {}

    for one in base_model_list:
        w[one] = base_model_list[one].state_dict()

    keys_list = {}
    key_set = set()
    tep_dict = {}
    for one in w:
        keys_list[one] = w[one].keys()
        key_set = set.union(key_set, list(w[one].keys()))

    for i in key_set:
        tep_dict[i] = []

    for i in key_set:
        for j in w:
            if i in w[j]:
                tep_dict[i].append(w[j][i])

    for i in tep_dict:
        tep_tep_dict = {}
        for tensor in tep_dict[i]:
            if tensor.shape not in tep_tep_dict:
                tep_tep_dict[tensor.shape] = [tensor.float()]
            else:
                tep_tep_dict[tensor.shape].append(tensor.float())
        for shape in tep_tep_dict:
            tep_tep_dict[shape] = torch.mean(torch.stack(tep_tep_dict[shape]),0)
        tep_dict[i] = tep_tep_dict

    for user in w:
        for key in w[user]:
            shape = w[user][key].shape
            w[user][key] = tep_dict[key][shape]

    for user in base_model_list:
        base_model_list[one].load_state_dict(w[user])

    return base_model_list



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


def Logits_aggregation_func(
    xloader,
    network_list,
    optimizer_list,
    logger,
    alignment_epoch
):
    for epoch in range(alignment_epoch):
        for i ,(inputs, targets) in enumerate(xloader):
            logits_list = {}
            average_logits = 0
            for user in network_list:
                network_list[user].train()
                optimizer_list[user].zero_grad()
                features, logits = network_list[user](inputs.to('cuda'))
                logits_list[user] = logits[0]
                average_logits += logits_list[user].clone().detach()

            average_logits = average_logits.div(float(len(network_list)))

            for user in network_list:
                criterion = torch.nn.L1Loss()
                loss = criterion(logits_list[user], average_logits)
                # logger.log("The Aggregation Loss of user {} is : {}".format(user, loss))
                loss.backward()
                optimizer_list[user].step()

    return None

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
    user_data = np.load('../../exps/NAS-Bench-201-algos/Dirichlet_0.1_Use_valid_{}_{}_non_iid_setting.npy'.format(valid_use, args.dataset), allow_pickle=True).item()
    train_loader_list = {}
    valid_loader_list = {}
    # alignment_loader = torch.utils.data.DataLoader(
    #     DatasetSplit(train_data, np.random.choice(list(range(len(train_data))), 5000)),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=True,
    # )

    alignment_loader = torch.utils.data.DataLoader(
        DatasetSplit(train_data, user_data['public']),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    user_num = len(user_data)-1

    for user in range(user_num):
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
        import ast
        import re

        file_proposal1 = '../../exps/NAS-Bench-201-algos/FedNAS_Search_darts.log'
        file_proposal = '../../exps/NAS-Bench-201-algos/Ours_Search_darts.log'
        # file_proposal = '../../exps/NAS-Bench-201-algos/FedNAS_128.log'

        file_proposal = args.extra_model_path
        genotype_list = {}

        if args.extra_model_path in Networks:
            for user in range(user_num):
                genotype_list[user] = Networks[args.extra_model_path]
        else:
            user_list = {}
            user = 0
            for line in open(file_proposal):
                if "<<<--->>>" in line:
                    tep_dict = ast.literal_eval(re.search('({.+})', line).group(0))
                    count = 0
                    for j in tep_dict['normal']:
                        for k in j:
                            if 'skip_connect' in k[0]:
                                count += 1
                    if count == 2:
                        genotype_list[user % 5] = tep_dict
                        user_list[user % 5] = user / 5
                    user += 1

        logger.log(genotype_list)

        base_model_list = {}
        for user in range(user_num):
            base_model_list[user] = obtain_model(model_config, genotype_list[user])
            flop, param = get_model_infos(base_model_list[user], xshape)
            logger.log("The model of User {}: parm: {}, Flops: {}.".format(user, param, flop))
            wandb.watch(base_model_list[user])

        # base_model = obtain_model(model_config, args.extra_model_path)
    elif args.model_source == "Densenet":
        base_model_list = {}
        for user in range(user_num):
            base_model_list[user] = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
            flop, param = get_model_infos(base_model_list[user], xshape)
            logger.log("The model of User {}: parm: {}, Flops: {}.".format(user, param, flop))
    else:
        base_model_list = {}
        for user in range(user_num):
            base_model_list[user], _, __ = create_cnn_model(args.model_source, args.dataset, optim_config.epochs + optim_config.warmup, None, use_cuda=1)
            flop, param = get_model_infos(base_model_list[user], xshape)
            logger.log("The model of User {}: parm: {}, Flops: {}.".format(user, param, flop))


        # raise ValueError("invalid model-source : {:}".format(args.model_source))


    optimizer_list = {}
    scheduler_list = {}
    criterion_list = {}
    for user in range(user_num):
        flop, param = get_model_infos(base_model_list[user], xshape)
        # logger.log("model ====>>>>:\n{:}".format(base_model_list[user]))
        # logger.log("model information : {:}".format(base_model_list[user].get_message()))
        logger.log("-" * 50)
        logger.log(
            "Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G".format(
                param, flop, flop / 1e3
            )
        )
        logger.log("-" * 50)
        optimizer_list[user], scheduler_list[user], criterion_list[user] = get_optim_scheduler(base_model_list[user].parameters(), optim_config)

        # logger.log("User{}, train_data : {:}".format(user, train_data[user]))
        # logger.log("User{}, valid_data : {:}".format(user, valid_data[user]))
        # optimizer, scheduler, criterion = get_optim_scheduler(
        #     base_model.parameters(), optim_config
        # )
        logger.log("User{}, optimizer  : {:}".format(user, optimizer_list[user]))
        logger.log("User{}, scheduler  : {:}".format(user, scheduler_list[user]))
        logger.log("User{}, criterion  : {:}".format(user, criterion_list[user]))
        # base_model_list[user], criterion_list[user] = torch.nn.DataParallel(base_model[user]).cuda(), criterion_list[user].cuda()
        criterion_list[user] = criterion_list[user].cuda()
        base_model_list[user] = base_model_list[user].cuda()

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )



    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start".format(last_info)
        )
        last_infox = torch.load(last_info)
        start_epoch = last_infox["epoch"] + 1
        last_checkpoint_path = last_infox["last_checkpoint"]
        if not last_checkpoint_path.exists():
            logger.log(
                "Does not find {:}, try another path".format(last_checkpoint_path)
            )
            last_checkpoint_path = (
                last_info.parent
                / last_checkpoint_path.parent.name
                / last_checkpoint_path.name
            )
        checkpoint = torch.load(last_checkpoint_path)

        for user in base_model_list:
            base_model_list[user].load_state_dict(checkpoint["model_{}".format(user)])
            optimizer_list[user].load_state_dict(checkpoint["optimizer_{}".format(user)])
            scheduler_list[user].load_state_dict(checkpoint["scheduler_{}".format(user)])
        valid_accuracies = checkpoint["valid_accuracies"]
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
                last_info, start_epoch
            )
        )
        del(checkpoint)
    elif args.resume is not None:
        assert Path(args.resume).exists(), "Can not find the resume file : {:}".format(
            args.resume
        )
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint["epoch"] + 1
        for user in base_model_list:
            base_model_list[user].load_state_dict(checkpoint["model_{}".format(user)])
            optimizer_list[user].load_state_dict(checkpoint["optimizer_{}".format(user)])
            scheduler_list[user].load_state_dict(checkpoint["scheduler_{}".format(user)])
        valid_accuracies = checkpoint["valid_accuracies"]
        logger.log(
            "=> loading checkpoint from '{:}' start with {:}-th epoch.".format(
                args.resume, start_epoch
            )
        )
    # elif args.init_model is not None:
    #     assert Path(
    #         args.init_model
    #     ).exists(), "Can not find the initialization file : {:}".format(args.init_model)
    #     checkpoint = torch.load(args.init_model)
    #     base_model.load_state_dict(checkpoint["base-model"])
    #     start_epoch, valid_accuracies, max_bytes = 0, {"best": -1}, {}
    #     logger.log("=> initialize the model from {:}".format(args.init_model))
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, max_bytes = 0, {"best": -1}, {}
    train_func, valid_func = get_procedures(args.procedure)
    total_epoch = optim_config.epochs + optim_config.warmup
    local_epoch = args.local_epoch
    # Main Training and Evaluation Loop
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(start_epoch, total_epoch):

        if args.logits_aggregation:
            Logits_aggregation_func(alignment_loader, base_model_list, optimizer_list, logger, 3)

        else:
            base_model_list = partial_average_weights(base_model_list)

        for user in scheduler_list:
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

        test_accuracy1_list = []
        test_accuracy5_list = []

        for user in train_loader_list:
            train_loss, train_acc1, train_acc5 = train_func(
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

            test_accuracy1_list.append(valid_acc1)
            test_accuracy5_list.append(valid_acc5)
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


        if np.average(test_accuracy1_list) > valid_accuracies["best"]:
            valid_accuracies["best"] = np.average(test_accuracy1_list)
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

        valid_accuracies[epoch] = np.average(test_accuracy1_list)
        info_dict = {
                     "average_valid_top1_acc": np.average(test_accuracy1_list),
                     "average_valid_top5_acc": np.average(test_accuracy5_list),
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


        checkpoint_dict = {
                "epoch": epoch,
                "args": deepcopy(args),
                "FLOP": flop,
                "PARAM": param,
                "model_source":args.model_source,
                "valid_accuracies": deepcopy(valid_accuracies),
                "model-config": model_config._asdict(),
                "optim-config": optim_config._asdict()
        }
        for user in base_model_list:
            checkpoint_dict["model_{}".format(user)] = base_model_list[user].state_dict()
            checkpoint_dict["scheduler_{}".format(user)] = scheduler_list[user].state_dict()
            checkpoint_dict["optimizer_{}".format(user)] = optimizer_list[user].state_dict()


        save_path = save_checkpoint(checkpoint_dict, model_base_path, logger)

        del(checkpoint_dict)

        if find_best:
            copy_checkpoint(model_base_path, model_best_path, logger)

        last_info = save_checkpoint(
            {
                "epoch": epoch,
                "args": deepcopy(args),
                "last_checkpoint": save_path,
            },
            logger.path("info"),
            logger,
        )

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
        self.local_epoch = 3
        self.datapath = '../../../data/{}'.format(self.dataset)
        self.model_source = 'autodl-searched'
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            base = 'CIFAR'
        self.model_config = './NAS-{}-none.config'.format(base)
        self.optim_config = './NAS-{}.config'.format(base)
        self.extra_model_path = 'GDAS_V1'
        self.procedure = 'basic'
        self.save_dir = './output/nas-infer/{}-BS{}-{}'.format(self.dataset, self.batch, self.extra_model_path)
        self.cut_out_length = 16
        self.workers = 4
        self.seed = 666
        self.print_freq = 500
        self.print_freq_eval = 1000
        self.logits_aggregation = False
        self.personalization_methods = "Fedavg"
        self.wandb_project = "Dirichlet_Federated_NAS_inference"
        # self.run_name = "{}-{}".format(self.model_source, self.dataset)
        self.run_name = "{}-{}-{}".format(self.extra_model_path, self.personalization_methods, self.dataset)
        self.resume_str = None


if __name__ == "__main__":

    # torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
    config = Config()
    import wandb
    wandb.init(project=config.wandb_project, name=config.run_name, resume = config.resume_str)
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
    parser.add_argument(
        "--logits_aggregation", type=bool, default=config.logits_aggregation, help="Batch size for training."
    )
    # Optimization options
    parser.add_argument(
        "--batch_size", type=int, default=config.batch, help="Batch size for training."
    )
    parser.add_argument(
        "--local_epoch", type=int, default=config.local_epoch, help="Local Epoch"
    )
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    assert args.save_dir is not None, "save-path argument can not be None"
    np.random.seed(args.rand_seed)
    wandb.config.update(args)
    main(args)

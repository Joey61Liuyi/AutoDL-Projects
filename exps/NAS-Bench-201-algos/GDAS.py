##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
###########################################################################
# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019 #
###########################################################################
import sys, time, random, argparse
from copy import deepcopy

import numpy as np
import torch

import sys
sys.path.append("../..")

from xautodl.config_utils import load_config, dict2config
from xautodl.datasets import get_datasets, get_nas_search_loaders
from xautodl.procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from xautodl.utils import get_model_infos, obtain_accuracy
from xautodl.log_utils import AverageMeter, time_string, convert_secs2time
from xautodl.models import get_cell_based_tiny_net, get_search_spaces
from nas_201_api import NASBench201API as API
from torch.utils.data import Dataset
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from dloptimizer import dlOptimizer
import copy
import warnings
warnings.filterwarnings("ignore")


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
        return torch.tensor(image), torch.tensor(label)

def average_weights(w, arch_personalization):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    arch_result = {}
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key].float(), len(w))
    # result = [copy.deepcopy(w_avg) for _ in range(len(w))]
    if arch_personalization:
        for i in range(0, len(w)):
            arch_result[i] = w[i]['arch_parameters']

    return w_avg, arch_result

def search_func(
    xloader,
    network,
    global_network,
    criterion,
    scheduler,
    w_optimizer,
    a_optimizer,
    epoch_str,
    print_freq,
    logger,
    local_epoch
):
    # network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.train()
    end = time.time()

    for _ in range(local_epoch):
        for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(
            xloader
        ):
            scheduler.update(None, 1.0 * step / len(xloader))
            base_targets = base_targets.cuda(non_blocking=True)
            arch_targets = arch_targets.cuda(non_blocking=True)
            # measure data loading time
            data_time.update(time.time() - end)

            # update the weights
            w_optimizer.zero_grad()
            _, logits = network(base_inputs.cuda())
            base_loss = criterion(logits, base_targets)
            base_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5)

            if args.baseline == 'dl':
                w_optimizer.step(global_network.get_weights())
            else:
                w_optimizer.step()
            # record
            base_prec1, base_prec5 = obtain_accuracy(
                logits.data, base_targets.data, topk=(1, 5)
            )
            base_losses.update(base_loss.item(), base_inputs.size(0))
            base_top1.update(base_prec1.item(), base_inputs.size(0))
            base_top5.update(base_prec5.item(), base_inputs.size(0))

            # update the architecture-weight
            a_optimizer.zero_grad()
            _, logits = network(arch_inputs.cuda())
            arch_loss = criterion(logits, arch_targets)
            arch_loss.backward()
            a_optimizer.step()
            # record
            arch_prec1, arch_prec5 = obtain_accuracy(
                logits.data, arch_targets.data, topk=(1, 5)
            )
            arch_losses.update(arch_loss.item(), arch_inputs.size(0))
            arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
            arch_top5.update(arch_prec5.item(), arch_inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % print_freq == 0 or step + 1 == len(xloader):
                Sstr = (
                    "*SEARCH* "
                    + time_string()
                    + " [{:}][{:03d}/{:03d}]".format(epoch_str, step, len(xloader))
                )
                Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
                    batch_time=batch_time, data_time=data_time
                )
                Wstr = "Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                    loss=base_losses, top1=base_top1, top5=base_top5
                )
                Astr = "Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                    loss=arch_losses, top1=arch_top1, top5=arch_top5
                )
                logger.log(Sstr + " " + Tstr + " " + Wstr + " " + Astr)
    return (
        base_losses.avg,
        base_top1.avg,
        base_top5.avg,
        arch_losses.avg,
        arch_top1.avg,
        arch_top5.avg,
        network.state_dict()
    )

def test_func(
    xloader,
    network,
    criterion,
):
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    network.eval()

    for step, (base_inputs, base_targets) in enumerate(
        xloader
    ):
        base_targets = base_targets.cuda(non_blocking=True)
        _, logits = network(base_inputs.cuda())
        base_loss = criterion(logits, base_targets)
        base_prec1, base_prec5 = obtain_accuracy(
            logits.data, base_targets.data, topk=(1, 5)
        )
        base_losses.update(base_loss.item(), base_inputs.size(0))
        base_top1.update(base_prec1.item(), base_inputs.size(0))
        base_top5.update(base_prec5.item(), base_inputs.size(0))

    return (
        base_losses.avg,
        base_top1.avg,
        base_top5.avg,
    )


def main(xargs):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    train_data, valid_data, xshape, class_num = get_datasets(
        xargs.dataset, xargs.data_path, -1
    )
    # config_path = 'configs/nas-benchmark/algos/GDAS.config'
    config = load_config(
        xargs.config_path, {"class_num": class_num, "xshape": xshape}, logger
    )
    search_loader, _, valid_loader = get_nas_search_loaders(
        train_data,
        valid_data,
        xargs.dataset,
        "../../configs/nas-benchmark/",
        config.batch_size,
        xargs.workers,
    )
    logger.log(
        "||||||| {:10s} ||||||| Search-Loader-Num={:}, batch size={:}".format(
            xargs.dataset, len(search_loader), config.batch_size
        )
    )
    logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))

    search_space = get_search_spaces("cell", xargs.search_space_name)
    if xargs.model_config is None:
        model_config = dict2config(
            {
                "name": "GDAS",
                "C": xargs.channel,
                "N": xargs.num_cells,
                "max_nodes": xargs.max_nodes,
                "num_classes": class_num,
                "space": search_space,
                "affine": False,
                "track_running_stats": bool(xargs.track_running_stats),
            },
            None,
        )
    else:
        model_config = load_config(
            xargs.model_config,
            {
                "num_classes": class_num,
                "space": search_space,
                "affine": False,
                "track_running_stats": bool(xargs.track_running_stats),
            },
            None,
        )

    search_model = {}
    w_optimizer = {}
    a_optimizer = {}
    w_scheduler = {}
    a_scheduler = {}
    valid_accuracies, genotypes = {}, {}

    search_globle_model = get_cell_based_tiny_net(model_config).cuda()
    for one in search_loader:
        search_model[one] = get_cell_based_tiny_net(model_config).cuda()
        search_model[one].load_state_dict(search_globle_model.state_dict())
        w_optimizer[one], w_scheduler[one], criterion = get_optim_scheduler(search_model[one].parameters(), config)
        if args.baseline == "dl":
            w_optimizer[one] = dlOptimizer(search_model[one].get_weights(), xargs.arch_learning_rate, 0.1)
        a_optimizer[one] = torch.optim.Adam(search_model[one].get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay,)
        valid_accuracies[one], genotypes[one] = (
            {"best": -1},
            {-1: search_model[one].genotype()},
        )


    criterion = criterion.cuda()
    logger.log("search-model :\n{:}".format(search_globle_model))
    logger.log("model-config : {:}".format(model_config))

    # logger.log("w-optimizer : {:}".format(w_optimizer))
    # logger.log("a-optimizer : {:}".format(a_optimizer))
    # logger.log("w-scheduler : {:}".format(w_scheduler))
    # logger.log("criterion   : {:}".format(criterion))
    flop, param = get_model_infos(search_globle_model, xshape)
    logger.log("FLOP = {:.2f} M, Params = {:.2f} MB".format(flop, param))
    logger.log("search-space [{:} ops] : {:}".format(len(search_space), search_space))
    if xargs.arch_nas_dataset is None:
        api = None
    else:
        api = API(xargs.arch_nas_dataset)
    logger.log("{:} create API = {:} done".format(time_string(), api))

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )

    # if last_info.exists():  # automatically resume from previous checkpoint
    #     logger.log(
    #         "=> loading checkpoint of the last-info '{:}' start".format(last_info)
    #     )
    #     last_info = torch.load(last_info)
    #     start_epoch = last_info["epoch"]
    #     checkpoint = torch.load(last_info["last_checkpoint"])
    #     genotypes = checkpoint["genotypes"]
    #     valid_accuracies = checkpoint["valid_accuracies"]
    #     search_model.load_state_dict(checkpoint["search_model"])
    #     w_scheduler.load_state_dict(checkpoint["w_scheduler"])
    #     w_optimizer.load_state_dict(checkpoint["w_optimizer"])
    #     a_optimizer.load_state_dict(checkpoint["a_optimizer"])
    #     logger.log(
    #         "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
    #             last_info, start_epoch
    #         )
    #     )
    # else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch = 0


    # start training
    start_time, search_time, epoch_time, total_epoch = (
        time.time(),
        AverageMeter(),
        AverageMeter(),
        config.epochs + config.warmup,
    )
    local_epoch = args.local_epoch
    for epoch in range(start_epoch, total_epoch):

        for user in w_scheduler:
            w_scheduler[user].update(epoch, 0.0)
            search_model[user].set_tau(
                xargs.tau_max - (xargs.tau_max - xargs.tau_min) * epoch / (total_epoch - 1)
            )

        # need_time = "Time Left: {:}".format(
        #     convert_secs2time(epoch_time.val * (total_epoch - epoch), True)
        # )
        epoch_str = "{:03d}-{:03d}".format(epoch, total_epoch)

        # logger.log(
        #     "\n[Search the {:}-th epoch] {:}, tau={:}, LR={:}".format(
        #         epoch_str, need_time, search_model.get_tau(), min(w_scheduler.get_lr())
        #     )
        # )
        weight_list = []
        acc_list = []
        test_acc_list = []
        for user in search_loader:
            (   search_w_loss,
                search_w_top1,
                search_w_top5,
                valid_a_loss,
                valid_a_top1,
                valid_a_top5,
                weight
            ) = search_func(
                search_loader[user],
                search_model[user],
                search_globle_model,
                criterion,
                w_scheduler[user],
                w_optimizer[user],
                a_optimizer[user],
                epoch_str,
                xargs.print_freq,
                logger,
                local_epoch
            )

            logger.log(
                "User {} : [{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s".format(
                    user, epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum
                )
            )
            logger.log(
                "User {} : [{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%".format(
                    user, epoch_str, valid_a_loss, valid_a_top1, valid_a_top5
                )
            )

            weight_list.append(weight)
            acc_list.append(valid_a_top1)

            valid_accuracies[user][epoch] = valid_a_top1
            genotypes[user][epoch] = search_model[user].genotype()

            # loss, top1acc, top5acc = test_func(valid_loader[user], search_model[user], criterion)
            # test_acc_list.append(top1acc)

            # logger.log(
            #     "||||---|||| The {epoch:}-th epoch, user {user}, valid loss={loss:.3f}, valid_top1={top1:.2f}%, valid_top5={top5:.2f}%".format(
            #         epoch=epoch_str, user=user, loss=loss, top1=top1acc, top5=top5acc, )
            # )


            info_dict = {
                         "{}user_w_loss".format(user): search_w_loss,
                         "{}user_w_top1".format(user): search_w_top1,
                         "{}user_w_top5".format(user): search_w_top5,
                         "{}user_a_loss".format(user): valid_a_loss,
                         "{}user_a_top1".format(user): valid_a_top1,
                         "{}user_a_top5".format(user): valid_a_top5,
                         # "{}user_test_loss".format(user): search_w_loss,
                         # "{}user_test_top1".format(user): search_w_loss,
                         # "{}user_test_top5".format(user): search_w_loss,
                         }
            wandb.log(info_dict)

        info_dict = {
            "epoch": epoch,
            "average_valid_acc": np.average(acc_list),
            "average_test_acc": np.average(test_acc_list)
        }
        wandb.log(info_dict)

        arch_personalize = args.personalize_arch
        weight_average, arch_list = average_weights(weight_list, arch_personalize)

        for user in search_model:
            if arch_personalize:
                tep = copy.deepcopy(weight_average)
                tep['arch_parameters'] = arch_list[user]
                search_model[user].load_state_dict(tep)
            else:
                search_model[user].load_state_dict(weight_average)

            logger.log(
                "<<<--->>> The {:}-th epoch : {:}".format(epoch_str, search_model[user].genotype())
            )
        search_globle_model.load_state_dict(weight_average)

        search_time.update(time.time() - start_time)

        # check the best accuracy

        # if valid_a_top1 > valid_accuracies["best"]:
        #     valid_accuracies["best"] = valid_a_top1
        #     genotypes["best"] = search_model.genotype()
        #     find_best = True
        # else:
        #     find_best = False


        # save checkpoint
        # save_path = save_checkpoint(
        #     {
        #         "epoch": epoch + 1,
        #         "args": deepcopy(xargs),
        #         "search_model": search_model.state_dict(),
        #         "w_optimizer": w_optimizer.state_dict(),
        #         "a_optimizer": a_optimizer.state_dict(),
        #         "w_scheduler": w_scheduler.state_dict(),
        #         "genotypes": genotypes,
        #         "valid_accuracies": valid_accuracies,
        #     },
        #     model_base_path,
        #     logger,
        # )
        # last_info = save_checkpoint(
        #     {
        #         "epoch": epoch + 1,
        #         "args": deepcopy(args),
        #         "last_checkpoint": save_path,
        #     },
        #     logger.path("info"),
        #     logger,
        # )
        # if find_best:
        #     logger.log(
        #         "<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.".format(
        #             epoch_str, valid_a_top1
        #         )
        #     )
        #     copy_checkpoint(model_base_path, model_best_path, logger)
        # with torch.no_grad():
        #     logger.log("{:}".format(search_globle_model.show_alphas()))
        # if api is not None:
        #     logger.log("{:}".format(api.query_by_arch(genotypes[epoch], "200")))
        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    # save checkpoint

    for user in search_model:

        model_base_path = logger.model_dir / "User{:}-acc-{}-basic-seed-{:}.pth".format(user, valid_accuracies[user][epoch],args.rand_seed)

        save_path = save_checkpoint(
            {
                "epoch": epoch + 1,
                "args": deepcopy(xargs),
                "search_model": search_model[user].state_dict(),
                "w_optimizer": w_optimizer[user].state_dict(),
                "a_optimizer": a_optimizer[user].state_dict(),
                "w_scheduler": w_scheduler[user].state_dict(),
                "genotypes": search_model[user].genotype(),
                "valid_accuracies": valid_accuracies[user],
            },
            model_base_path,
            logger,

        )

    # logger.log("\n" + "-" * 100)
    # # check the performance from the architecture dataset
    # logger.log(
    #     "GDAS : run {:} epochs, cost {:.1f} s, last-geno is {:}.".format(
    #         total_epoch, search_time.sum, genotypes[total_epoch - 1]
    #     )
    # )
    # if api is not None:
    #     logger.log("{:}".format(api.query_by_arch(genotypes[total_epoch - 1], "200")))

    logger.close()


if __name__ == "__main__":

    import wandb
    wandb.init(project="Federated_NAS", name='cifar10_Ours')
    dataset = 'cifar10'
    space = 'darts'
    track_running_stats = 1

    parser = argparse.ArgumentParser("GDAS")
    parser.add_argument("--data_path", type=str, default= '../../../data/{}'.format(dataset),help="The path to dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default = dataset,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Choose between Cifar10/100 and ImageNet-16.",
    )
    # channels and number-of-cells
    parser.add_argument("--search_space_name", type=str, default=space,help="The search space name.")
    parser.add_argument("--max_nodes", type=int, default=4, help="The maximum number of nodes.")
    parser.add_argument("--channel", type=int, default=16, help="The number of channels.")
    parser.add_argument(
        "--num_cells", type=int, default=2, help="The number of cells in one stage."
    )
    parser.add_argument(
        "--track_running_stats",
        type=int,
        default = track_running_stats,
        choices=[0, 1],
        help="Whether use track_running_stats or not in the BN layer.",
    )
    parser.add_argument(
        "--config_path", type=str, default= '../../configs\search-opts\GDAS-NASNet-CIFAR.config',help="The path of the configuration."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default='../../configs\search-archs\GDASFRC-NASNet-CIFAR.config',
        help="The path of the model configuration. When this arg is set, it will cover max_nodes / channels / num_cells.",
    )
    # architecture leraning rate
    parser.add_argument(
        "--arch_learning_rate",
        type=float,
        default=3e-4,
        help="learning rate for arch encoding",
    )
    parser.add_argument(
        "--arch_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for arch encoding",
    )
    parser.add_argument("--tau_min", type=float, default=0.1, help="The minimum tau for Gumbel")
    parser.add_argument("--tau_max", type=float, default=10, help="The maximum tau for Gumbel")
    # log
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--save_dir", type=str, default = './output/search-cell-{}/GDAS-FRC-{}-BN{}'.format(space, dataset, track_running_stats), help="Folder to save checkpoints and log."
    )
    parser.add_argument(
        "--arch_nas_dataset",
        default=None,
        type=str,
        help="The path to load the architecture dataset (tiny-nas-benchmark).",
    )
    parser.add_argument("--print_freq", type=int, default=200, help="print frequency (default: 200)")
    parser.add_argument("--local_epoch", type=int, default=5, help="local_epochs for edge nodes")
    parser.add_argument("--personalize_arch", type=bool, default=True, help="local_epochs for edge nodes")
    parser.add_argument("--non_iid_level", type = float, default= 0.5, help="non_iid level settings")
    parser.add_argument("--baseline", type =str, default = None, help = "type of baseline")
    parser.add_argument("--rand_seed", type=int, default=61, help="manual seed")
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)



    wandb.config.update(args)
    main(args)

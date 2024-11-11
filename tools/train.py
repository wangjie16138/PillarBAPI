import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    # 预训练模型
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    # 是否使用同步批量归一化（Batch Normalization）。
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    # 评价模型的轮数间隔
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    # 通过命令行参数和配置文件设置的配置信息
    return args, cfg


def main():
    # 解析命令行参数，和配置文件
    args, cfg = parse_config()
    # 是否分布式训练
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        # 获取配置文件的参数配置
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        # 确保每个GPU都能获得等量的数据批次
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus
    # 轮数设置
    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)
    """
     代码首先构建了输出目录output_dir的路径，这个路径是由多个子目录拼接而成，包括根目录cfg.ROOT_DIR、
     固定的output目录、实验组路径cfg.EXP_GROUP_PATH、标签cfg.TAG和额外的标签args.extra_tag。然后，
     创建了一个名为ckpt的子目录，用于存放训练过程中的检查点（checkpoints）。mkdir方法用于创建目录，
     parents=True表示如果父目录不存在也一并创建，exist_ok=True表示如果目录已存在则不抛出异常。
     """
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # 创建日志文件名，创建日志记录器
    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    # 获取GPU列表
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    # 记录训练模式
    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')
    # 记录命令行参数
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    # 记录配置文件内容
    log_config_to_file(cfg, logger=logger)
    # 复制配置文件到输出目录
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))
    # 初始化TensorBoard日志记录器 可视化
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info("----------- Create dataloader & network & optimizer -----------")
    # 构建数据加载器
    """
    dataset_cfg=cfg.DATA_CONFIG：数据集的配置信息，可能包括数据集的路径、数据预处理方式等。
    class_names=cfg.CLASS_NAMES：类别的名称列表，用于数据集的分类任务。
    batch_size=args.batch_size：每个批次的数据量大小。
    dist=dist_train：一个布尔值，表示是否进行分布式训练。
    workers=args.workers：用于数据加载的工作进程数。
    logger=logger：日志记录器对象，用于在构建数据加载器的过程中记录日志。
    training=True：一个布尔值，指示正在构建的是训练用的数据加载器。
    merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch：一个布尔值，如果为 True，则将所有迭代合并到一个epoch中。
    total_epochs=args.epochs：总的训练周期数。
    seed=666 if args.fix_random_seed else None：随机数种子。如果 args.fix_random_seed 为 True，则使用种子值 666 来固定随机数生成器的状态，以确保实验的可重复性；否则，使用 None 表示不固定随机数种子。
    返回三个主要对象：训练数据集（train_set）、训练数据加载器（train_loader）以及训练数据采样器（train_sampler）
    """
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )
    # 网络模型构建 优化器构建
    """
    model_cfg=cfg.MODEL：模型的配置信息，可能包括网络结构、层数、参数等。
    num_class=len(cfg.CLASS_NAMES)：类别的数量，这里通过计算类别名称列表的长度来确定。
    dataset=train_set：训练数据集对象，这个参数可能被用来为网络提供一些关于数据集的额外信息，例如输入数据的形状、预处理方式等
    """
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    """
    这段代码检查 args.sync_bn 是否为 True。如果是，它会使用 torch.nn.SyncBatchNorm.convert_sync_batchnorm
     方法将模型中的所有 BatchNorm 层转换为同步批归一化（SyncBatchNorm）层。同步批归一化通常用于分布式训练，
     以确保不同GPU上的模型在更新时能够同步归一化统计信息
    """
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # 这行代码将模型移动到GPU上
    model.cuda()
    """
    调用了一个名为 build_optimizer 的函数来创建优化器。优化器用于在训练过程中更新模型的权重。
    这个函数接收两个参数：
    model：前面构建的模型，优化器会根据模型的参数来创建。
    cfg.OPTIMIZATION：优化器的配置信息，可能包括优化器的类型（如SGD、Adam等）、学习率、动量等。
    """
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    """
     这段代码的主要目的是在训练开始前，尽可能地加载最新的模型参数和优化器状态，
     以便从之前的训练状态继续训练。如果没有提供特定的检查点文件路径，
     代码会尝试从预设的目录中找到最新的检查点文件来加载。
    """
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
    # 检查点文件
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        # 使用glob模块来查找ckpt_dir目录下所有以.pth结尾的文件，并将它们放入ckpt_list列表中
        ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
              
        if len(ckpt_list) > 0:
            # 对ckpt_list列表进行排序，排序的依据是检查点文件的修改时间（最新的文件排在最前面）
            ckpt_list.sort(key=os.path.getmtime)
            while len(ckpt_list) > 0:
                try:
                    it, start_epoch = model.load_params_with_optimizer(
                        ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                    )
                    last_epoch = start_epoch + 1
                    break
                except:
                    ckpt_list = ckpt_list[:-1]

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    # 分布式训练环境下设置模型
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    """
    model: 需要训练的神经网络模型。
    optimizer: 优化器，用于更新模型的权重。
    train_loader: 数据加载器，用于从训练数据集中加载数据。
    model_func=model_fn_decorator(): 可能是用于包装或修改模型行为的函数或装饰器。
    lr_scheduler: 学习率调度器，用于在训练过程中调整学习率。
    optim_cfg: 优化器的配置，可能包含学习率、动量等参数。
    start_epoch: 开始训练的周期数（例如，如果从检查点恢复训练）。
    total_epochs: 总共需要训练的周期数。
    start_iter: 开始训练的迭代次数（与周期数不同，一个周期可能包含多个迭代）。
    rank: 当前进程的排名，用于分布式训练。
    tb_log: TensorBoard 日志对象，用于可视化训练过程。
    ckpt_save_dir: 检查点保存目录，用于保存模型的中间状态。
    train_sampler: 训练数据的采样器，用于控制数据加载的顺序。
    lr_warmup_scheduler: 学习率预热调度器，用于在训练初期逐渐提高学习率。
    ckpt_save_interval: 保存检查点的间隔（以周期或迭代次数为单位）。
    max_ckpt_save_num: 最大保存的检查点数量，用于限制保存空间。
    merge_all_iters_to_one_epoch: 是否将所有迭代合并为一个周期，影响日志记录和检查点保存。
    logger: 日志记录器对象，用于记录训练过程中的信息。
    logger_iter_interval: 日志记录的迭代间隔。
    ckpt_save_time_interval: 检查点保存的时间间隔。
    use_logger_to_record: 是否使用日志记录器来记录信息，而不是其他方式（如 tqdm）。
    show_gpu_stat: 是否显示 GPU 状态信息。
    use_amp: 是否使用自动混合精度（Automatic Mixed Precision）训练，这可以加速训练并减少 GPU 内存使用。
    cfg: 配置对象，可能包含上述未明确列出的其他配置信息。
    """
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg
    )
    # 资源管理和清理
    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)  # Only evaluate the last args.num_epochs_to_eval epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()

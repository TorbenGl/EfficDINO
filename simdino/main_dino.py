# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import builtins
import os
import sys
import signal
import datetime
import time
import math
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.nn as dist_nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead
import wandb

# Global flag for graceful interruption
_INTERRUPTED = False


# -------- HuggingFace ImageNet dataset helpers --------

class HFImageNetDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace datasets.Dataset as a PyTorch Dataset."""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    @property
    def targets(self):
        return [int(x) for x in self.dataset['label']]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample['image']
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert('RGB')
        label = int(sample['label'])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _load_hf_dataset(data_path, split):
    import os
    from datasets import load_dataset
    cache_dir = os.path.join(data_path, ".hf_cache")
    return load_dataset(data_path, split=split, trust_remote_code=False, cache_dir=cache_dir)

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument('--z_dim', default=256, type=int, 
                        help="""Dimensionality of the DINO head bottleneck dim (default: 256).""")
    parser.add_argument('--hidden_dim', default=2048, type=int, 
                        help="""Dimensionality of the DINO head hidden dim (default: 2048).""")
    parser.add_argument('--use_simdino', default=True, type=utils.bool_flag,
        help="Whether to use sim dino method (Default: True)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--compile', type=utils.bool_flag, default=True, help="""Whether or not compile model.""")
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--dataset_type', default='imagefolder', type=str,
        choices=['imagefolder', 'hf_imagenet'],
        help='Dataset format: imagefolder (default, expects class subdirs) or hf_imagenet (HuggingFace snapshot_download root)')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--track_wandb", action='store_true', help='logging with wandb')
    parser.add_argument("--track_swan", action='store_true', help='log with swanlab')

    # MCR
    parser.add_argument('--coeff', type=float, default=1,
                        help='coefficient of cosine similarity (default: 1)')
    parser.add_argument('--eps', type=float, default=0.5,
                    help='eps for TCR (default: 0.5)')
    parser.add_argument('--reduce_cov', type=int, default=0, help="""Whether or not all_reduce covariance matrices across gpus.""")
    parser.add_argument('--expa_type', type=int, default=1, help="""Whether or not apply smoothing in expansion_term.""")

    # EfficEmbed
    parser.add_argument('--embed_type', default='standard', type=str, choices=['standard', 'efficembed'],
        help='Patch embedding type: standard (Conv2d) or efficembed (sub-patch tokenization)')
    parser.add_argument('--sub_patch_size', default=4, type=int,
        help='Sub-patch conv kernel size for efficembed (default: 4)')
    parser.add_argument('--sub_patch_channels', default=None, type=int,
        help='Number of channels for sub-patch conv (default: auto-computed to match embed_dim)')

    # Gradient accumulation
    parser.add_argument('--grad_accum_steps', default=1, type=int,
        help='Number of gradient accumulation steps (effective_batch = batch_size_per_gpu * world_size * grad_accum_steps)')

    # Enhanced checkpointing
    parser.add_argument('--keep_last_ckpts', default=3, type=int,
        help='Number of periodic checkpoints to keep (older ones are deleted)')

    # Periodic k-NN evaluation
    parser.add_argument('--eval_freq', default=0, type=int,
        help='Run k-NN evaluation every N epochs during training (0 to disable)')
    parser.add_argument('--eval_data_path', default='', type=str,
        help='Path to ImageNet root (with train/ and val/ subdirs) for periodic k-NN eval')

    # WandB
    parser.add_argument('--wandb_project', default='simdinov1', type=str, help='WandB project name')
    parser.add_argument('--wandb_run_name', default='', type=str, help='WandB run name (default: output_dir basename)')
    parser.add_argument('--wandb_run_id', default='', type=str,
        help='WandB run ID for resuming logging (auto-saved in checkpoint)')

    return parser


def _sigterm_handler(signum, frame):
    global _INTERRUPTED
    _INTERRUPTED = True
    print(f"\nReceived signal {signum}, will save checkpoint and exit after current epoch...")


def _build_checkpoint(student, teacher, optimizer, fp16_scaler, dino_loss, epoch, args,
                      best_knn=0.0, wandb_run_id=''):
    save_dict = {
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + 1,
        'args': args,
        'dino_loss': dino_loss.state_dict(),
        'best_knn': best_knn,
        'wandb_run_id': wandb_run_id,
        'rng_states': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all(),
        },
    }
    if fp16_scaler is not None:
        save_dict['fp16_scaler'] = fp16_scaler.state_dict()
    return save_dict


def _rotate_checkpoints(output_dir, keep_last=3):
    """Keep only the last N periodic checkpoints (checkpoint_NNNN.pth)."""
    import glob
    ckpts = sorted(glob.glob(os.path.join(output_dir, 'checkpoint_[0-9][0-9][0-9][0-9].pth')))
    # Never delete checkpoint_best.pth, checkpoint.pth, or checkpoint_interrupted.pth
    if len(ckpts) > keep_last:
        for old_ckpt in ckpts[:-keep_last]:
            os.remove(old_ckpt)
            print(f"Removed old checkpoint: {old_ckpt}")


def _run_knn_eval(model, args):
    """Run k-NN evaluation on ImageNet val set. Returns top1 accuracy."""
    from eval_knn import extract_features, knn_classifier, ReturnIndexDataset, ReturnIndexHFDataset
    from torchvision import transforms as pth_transforms

    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_type = getattr(args, 'dataset_type', 'imagefolder')
    if dataset_type == 'hf_imagenet':
        eval_root = args.eval_data_path or args.data_path
        dataset_train = ReturnIndexHFDataset(_load_hf_dataset(eval_root, 'train'), transform=transform)
        dataset_val = ReturnIndexHFDataset(_load_hf_dataset(eval_root, 'validation'), transform=transform)
    else:
        dataset_train = ReturnIndexDataset(os.path.join(args.eval_data_path, "train"), transform=transform)
        dataset_val = ReturnIndexDataset(os.path.join(args.eval_data_path, "val"), transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler, batch_size=128,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=128,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    model.eval()
    train_features = extract_features(model, data_loader_train, use_cuda=True)
    test_features = extract_features(model, data_loader_val, use_cuda=True)

    top1 = 0.0
    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)
        train_labels = torch.tensor(dataset_train.targets).long().cuda()
        test_labels = torch.tensor(dataset_val.targets).long().cuda()
        top1, top5 = knn_classifier(train_features.cuda(), train_labels,
                                    test_features.cuda(), test_labels, k=20, T=0.07)
        print(f"k-NN eval: Top1={top1:.2f}%, Top5={top5:.2f}%")

    # Broadcast result to all ranks
    top1_tensor = torch.tensor([top1], device='cuda')
    dist.broadcast(top1_tensor, src=0)
    model.train()
    return top1_tensor.item()


def train_dino(args):
    global _INTERRUPTED
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    # Register signal handlers for graceful interruption
    signal.signal(signal.SIGTERM, _sigterm_handler)
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, _sigterm_handler)

    if utils.is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir, "args.log"), "w") as f:
            f.write(" ".join(sys.argv) + "\n")
            f.write(str(vars(args)))
        import shutil
        shutil.copyfile(Path(__file__), f"{args.output_dir}/main.py")
    print_ = builtins.print
    log_file = Path(args.output_dir, "output.log")
    def print(*args, **kwargs):
        print_(*args, **kwargs)
        if 'file' not in kwargs:
            with open(log_file, "a") as f:
                print_(*args, **kwargs, file=f)
    builtins.print = print
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    effective_batch = args.batch_size_per_gpu * utils.get_world_size() * args.grad_accum_steps
    print(f"Effective batch size: {args.batch_size_per_gpu} x {utils.get_world_size()} GPUs x {args.grad_accum_steps} accum = {effective_batch}")

    assert not(args.track_wandb and args.track_swan), "Please do not use both tracking methods simultaneously."

    # ============ optionally resume training (load checkpoint first to get wandb_run_id) ============
    # We do a pre-check for resume to extract wandb_run_id before initializing wandb
    resumed_wandb_id = ''
    ckp_path = os.path.join(args.output_dir, "checkpoint.pth")
    if os.path.isfile(ckp_path):
        ckp = torch.load(ckp_path, map_location="cpu")
        resumed_wandb_id = ckp.get('wandb_run_id', '')
        del ckp

    # ============ setup wandb ============
    wandb_run_id = ''
    if args.track_wandb and utils.is_main_process():
        runname = args.wandb_run_name or args.output_dir.rstrip('/').split('/')[-1]
        # Resume wandb run if we have a run_id from checkpoint or CLI
        resume_id = args.wandb_run_id or resumed_wandb_id
        if resume_id:
            wandb.init(project=args.wandb_project, name=runname, id=resume_id, resume="allow")
        else:
            wandb.init(project=args.wandb_project, name=runname)
        wandb.config.update(args, allow_val_change=True)
        wandb_run_id = wandb.run.id

    # ============ setup swanlab ============
    if args.track_swan and utils.is_main_process():
        import swanlab
        runname = args.wandb_run_name or args.output_dir.rstrip('/').split('/')[-1]
        swanlab.sync_wandb()
        wandb.init(project=args.wandb_project, name=runname, mode='offline')
        wandb.config.update(args, allow_val_change=True)
    args.enable_logging = utils.is_main_process() and (args.track_wandb or args.track_swan)

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    if getattr(args, 'dataset_type', 'imagefolder') == 'hf_imagenet':
        dataset = HFImageNetDataset(_load_hf_dataset(args.data_path, 'train'), transform=transform)
    else:
        dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    args.arch = args.arch.replace("deit", "vit")
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,
            embed_type=args.embed_type,
            sub_patch_size=args.sub_patch_size,
            sub_patch_channels=args.sub_patch_channels,
        )
        teacher = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            embed_type=args.embed_type,
            sub_patch_size=args.sub_patch_size,
            sub_patch_channels=args.sub_patch_channels,
        )
        embed_dim = student.embed_dim
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.z_dim,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head, hidden_dim=args.hidden_dim, bottleneck_dim=args.z_dim),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    if args.compile:
        compile_backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
        teacher = torch.compile(teacher, backend=compile_backend)
        student = torch.compile(student, backend=compile_backend)
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Student and Teacher are built: they are both {args.arch} network (embed_type={args.embed_type}).")

    # ============ preparing loss ... ============
    if args.use_simdino:
        dino_loss = MCRLoss(
            args.local_crops_number + 2,
            args.reduce_cov,
            args.expa_type,
            args.eps,
            args.coeff,
        ).cuda()
    else:
        dino_loss = DINOLoss(
            args.out_dim,
            args.local_crops_number + 2,
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups, fused=True)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9, fused=True)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups, fused=True)
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    # Schedules are per-iteration. With grad_accum, one "optimizer step" = grad_accum_steps iterations.
    # The schedule length = epochs * iters_per_epoch (micro-batch iterations).
    # We index by global_step (optimizer step count), so schedule length = epochs * steps_per_epoch.
    steps_per_epoch = len(data_loader) // args.grad_accum_steps
    lr_schedule = utils.cosine_scheduler(
        args.lr * effective_batch / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, steps_per_epoch,
    )
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, steps_per_epoch)
    print(f"Loss, optimizer and schedulers ready. Steps per epoch: {steps_per_epoch}")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "best_knn": 0.0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]
    best_knn = to_restore.get("best_knn", 0.0)

    # Restore RNG states if available
    if os.path.isfile(ckp_path):
        ckp = torch.load(ckp_path, map_location="cpu")
        rng_states = ckp.get('rng_states', None)
        if rng_states is not None:
            random.setstate(rng_states['python'])
            np.random.set_state(rng_states['numpy'])
            torch.random.set_rng_state(rng_states['torch'])
            torch.cuda.set_rng_state_all(rng_states['cuda'])
            print("Restored RNG states from checkpoint.")
        del ckp

    start_time = time.time()
    print(f"Starting DINO training from epoch {start_epoch}!")
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        epoch_time = time.time() - epoch_start

        # ============ periodic k-NN evaluation ... ============
        knn_top1 = 0.0
        if (args.eval_freq > 0 and args.eval_data_path and
                (epoch % args.eval_freq == 0 or epoch == args.epochs - 1)):
            # Extract teacher backbone for eval
            teacher_backbone = teacher_without_ddp.backbone
            knn_top1 = _run_knn_eval(teacher_backbone, args)
            if knn_top1 > best_knn:
                best_knn = knn_top1
                print(f"New best k-NN: {best_knn:.2f}%")

        # ============ writing checkpoint ... ============
        save_dict = _build_checkpoint(student, teacher, optimizer, fp16_scaler,
                                       dino_loss, epoch, args, best_knn, wandb_run_id)

        # Always save latest checkpoint
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))

        # Periodic checkpoint with rotation
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint_{epoch:04d}.pth'))
            if utils.is_main_process():
                _rotate_checkpoints(args.output_dir, keep_last=args.keep_last_ckpts)

        # Save best checkpoint
        if knn_top1 > 0 and knn_top1 >= best_knn:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint_best.pth'))

        # ============ writing logs ... ============
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'epoch_time': epoch_time,
                     'best_knn': best_knn}
        if knn_top1 > 0:
            log_stats['knn_top1'] = knn_top1
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        if args.enable_logging:
            epoch_log = {"epoch": epoch, "epoch_time": epoch_time, "best_knn": best_knn}
            if knn_top1 > 0:
                epoch_log["knn_top1"] = knn_top1
            wandb.log(epoch_log)

        # ============ handle interruption ... ============
        if _INTERRUPTED:
            print(f"Interrupted at epoch {epoch}. Saving checkpoint_interrupted.pth ...")
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint_interrupted.pth'))
            print("Checkpoint saved. Exiting.")
            sys.exit(0)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.enable_logging:
        wandb.finish()


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,
                    epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    accum_steps = args.grad_accum_steps
    steps_per_epoch = len(data_loader) // accum_steps

    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # Determine if this is the last micro-step in an accumulation group
        micro_step = it % accum_steps
        is_accum_step = (micro_step == accum_steps - 1) or (it == len(data_loader) - 1)

        # Global optimizer step index (for schedule lookup)
        global_step = steps_per_epoch * epoch + (it // accum_steps)

        # Update LR and WD on the first micro-step of each accumulation group
        if micro_step == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[global_step]
                if i == 0:
                    param_group["weight_decay"] = wd_schedule[global_step]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        # Use no_sync for non-final accumulation steps (skip all-reduce)
        maybe_no_sync = student.no_sync if not is_accum_step else nullcontext
        with maybe_no_sync():
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                teacher_output = teacher(images[:2])
                student_output = student(images)
                if args.use_simdino:
                    loss, comp_loss, expa_loss = dino_loss(student_output, teacher_output)
                else:
                    loss = dino_loss(student_output, teacher_output, epoch)
                # Scale loss by accumulation steps
                loss = loss / accum_steps

            if not math.isfinite(loss.item() * accum_steps):
                print("Loss is {}, stopping training".format(loss.item() * accum_steps), force=True)
                sys.exit(1)

            # Backward
            if fp16_scaler is None:
                loss.backward()
            else:
                fp16_scaler.scale(loss).backward()

        # Only step optimizer on accumulation boundary
        if is_accum_step:
            param_norms = None
            if fp16_scaler is None:
                if args.clip_grad:
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
                optimizer.step()
            else:
                fp16_scaler.unscale_(optimizer)
                if args.clip_grad:
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            optimizer.zero_grad()

            # EMA update for the teacher (once per optimizer step)
            with torch.no_grad():
                m = momentum_schedule[global_step]
                if hasattr(torch, '_foreach_lerp_'):
                    torch._foreach_lerp_(list(teacher_without_ddp.parameters()),
                                         list(student.module.parameters()), weight=1. - m)
                else:
                    torch._foreach_mul_(list(teacher_without_ddp.parameters()), m)
                    torch._foreach_add_(list(teacher_without_ddp.parameters()),
                                        list(student.module.parameters()), alpha=1. - m)

            # logging (once per optimizer step)
            torch.cuda.synchronize()
            actual_loss = loss.item() * accum_steps  # unscaled loss
            metric_logger.update(loss=actual_loss)
            if args.use_simdino:
                metric_logger.update(expa_loss=expa_loss.item())
                metric_logger.update(comp_loss=comp_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

            if args.enable_logging:
                logs2wb = {
                    "loss": actual_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "wd": optimizer.param_groups[0]["weight_decay"],
                    "ema_momentum": m,
                    "gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
                }
                if args.use_simdino:
                    logs2wb["expa_loss"] = expa_loss.item()
                    logs2wb["comp_loss"] = comp_loss.item()
                if param_norms:
                    logs2wb["grad_norm"] = max(param_norms)
                wandb.log(logs2wb)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

class MCRLoss(nn.Module):
    def __init__(self, ncrops, reduce_cov=0, expa_type=0, eps=0.5, coeff=1.0):
        super().__init__()
        self.ncrops = ncrops
        self.eps = eps
        self.coeff = coeff
        self.reduce_cov = reduce_cov
        self.expa_type = expa_type

    def forward(self, student_feat, teacher_feat):
        """
        Expansion Loss and Compression Loss between features of the teacher and student networks.
        """
        student_feat = student_feat.view(self.ncrops, -1, student_feat.shape[-1])
        teacher_feat = teacher_feat.view(2, -1, teacher_feat.shape[-1])
        
        comp_loss = self.calc_compression(student_feat, teacher_feat)
        if self.expa_type == 0: # only compute expansion on global views
            expa_loss = self.calc_expansion(student_feat[:len(teacher_feat)])
        elif self.expa_type == 1:
            expa_loss = self.calc_expansion((student_feat[:len(teacher_feat)]+teacher_feat)/2)
        loss = - self.coeff * comp_loss - expa_loss
        return loss, comp_loss.detach(), expa_loss.detach()
    
    def calc_compression(self, student_feat_list, teacher_feat_list):
        """
        Compute compression loss between student and teacher features.
        """
        # Convert lists of tensors to a single tensor for vectorized operations
        
        sim = F.cosine_similarity(teacher_feat_list.unsqueeze(1), student_feat_list.unsqueeze(0), dim=-1)
        sim.view(-1, sim.shape[-1])[:: (len(student_feat_list) + 1), :].fill_(0)  # Trick to fill diagonal
        
        n_loss_terms = len(teacher_feat_list)* len(student_feat_list) - min(len(teacher_feat_list), len(student_feat_list))
        # Sum the cosine similarities
        comp_loss = sim.mean(2).sum()/n_loss_terms
        # global_comp_loss = (sim[:, :len(teacher_feat_list)].mean(2).sum()).detach_().div_(len(teacher_feat_list))
        return comp_loss
    
    def calc_expansion(self, feat_list) -> torch.Tensor:
        """
        Compute expansion loss using Coding Rate estimation.
        """
        cov_list = []
        num_views = len(feat_list)
        m, p = feat_list[0].shape
        
        cov_list = [W.T.matmul(W) for W in feat_list]
        cov_list = torch.stack(cov_list)
        N=1
        if dist.is_initialized():
            N = dist.get_world_size()
            if self.reduce_cov == 1:
                cov_list = dist_nn.all_reduce(cov_list)
        scalar = p / (m * N * self.eps)
        I = torch.eye(p, device=cov_list[0].device)
        loss:torch.Tensor = 0
        for i in range(num_views):
            loss += torch.linalg.cholesky_ex(I + scalar * cov_list[i])[0].diagonal().log().sum()
        loss /= num_views
        loss *= (p+N*m)/(p*N*m) # the balancing factor gamma, you can also use the next line. This is ultimately a heuristic, so feel free to experiment.
        # loss *= ((self.eps * N * m) ** 0.5 / p)
        return loss
    

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    train_dino(args)

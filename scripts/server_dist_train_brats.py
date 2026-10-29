import argparse
import datetime
import logging
import os
import random
import sys

sys.path.append(".")
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import brats as brats
from wetrPRWT.PAR import PAR
from utils import imutils
from utils.AverageMeter import AverageMeter
from utils.optimizer import PolyWarmupAdamW
from wetrPRWT.model_attn_aff import WeTr
import utils.pyutils as pyutils
import utils.camutils as camutils

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default=r'.../brats_attn_reg.yaml',
                    type=str,
                    help="config")
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(rank,filename='test.log'):
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if rank == 0:
        fHandler = logging.FileHandler(filename, mode='w')
        fHandler.setFormatter(logFormatter)
        logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def get_down_size(ori_shape=(512, 512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5
def sym_kl_with_confidence(segs_a, segs_b, T=0.5, conf_thr=0.6, eps=1e-6):
    # segs_*: [B, C, H, W] logits
    p = F.softmax(segs_a / T, dim=1)
    q = F.softmax(segs_b / T, dim=1)

    # 置信度：两边都要高才算
    conf = torch.max(p, dim=1)[0] * torch.max(q, dim=1)[0]  # [B,H,W]
    mask = (conf > conf_thr).float()

    # KL(p||q) 与 KL(q||p) 对称
    kl_pq = F.kl_div((p + eps).log(), q, reduction='none').sum(1)  # [B,H,W]
    kl_qp = F.kl_div((q + eps).log(), p, reduction='none').sum(1)

    loss = ((kl_pq + kl_qp) * mask).sum() / (mask.sum() + eps)
    return loss

def attn_back_from_rot(attn_rot, Ht, Wt, k, eps=1e-4):

    device = attn_rot.device
    N = Ht * Wt

    grid = torch.arange(N, device=device).reshape(Ht, Wt)            # orig index grid
    grid_rot = torch.rot90(grid, k=k, dims=[0, 1]).reshape(-1)        # rotated-order: j -> orig i
    inv = torch.argsort(grid_rot)                                    # orig i -> rotated-order j

    if attn_rot.dim() == 3:      # [B,N,N]
        return attn_rot[:, inv][:, :, inv]
    elif attn_rot.dim() == 4:    # [B,heads,N,N]
        return attn_rot[:, :, inv][:, :, :, inv]
    else:
        raise ValueError(f"Unexpected attn_rot shape: {attn_rot.shape}")


def train(rank, world_size, cfg):
    print('*********************************************************************************************************')
    print(rank, world_size)
    dist.init_process_group(backend=cfg.train.backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    setup_logger(filename=os.path.join(cfg.work_dir.tb_logger_dir, f'train_{rank}.log'), rank=rank)
    logging.info("Total gpus: %d, samples per gpu: %d..." % (dist.get_world_size(), cfg.train.samples_per_gpu))

    train_dataset = brats.brats2021ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )


    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              # shuffle=True,
                              num_workers=cfg.train.num_workers,
                              pin_memory=False,
                              drop_last=True,
                              sampler=train_sampler,
                              prefetch_factor=4)

    wetr = WeTr(backbone=cfg.backbone.config,
                stride=cfg.backbone.stride,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=True,
                pooling=cfg.backbone.pooling, )
    logging.info('\nNetwork config: \n%s' % (wetr))
    param_groups = wetr.get_param_groups()
    par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24])
    wetr.to(device)
    par.to(device)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,  ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)
    wetr = torch.nn.SyncBatchNorm.convert_sync_batchnorm(wetr)
    wetr = DistributedDataParallel(wetr, device_ids=[rank], find_unused_parameters=True)
    train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))

    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    list_meter = pyutils.ListMeter()
    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs, mask, cls_labels, img_box,_ = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, mask, cls_labels, img_box,_ = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_labels = cls_labels.to(device, non_blocking=True)


        cams = camutils.multi_scale_cam_with_aff_mat(wetr, inputs=inputs, scales=cfg.cam.scales)

        cls, segs, attns, attn_pred = wetr(inputs, seg_detach=cfg.train.seg_detach)

        # ----------------- Joint Rotation Consistency (JRC) -----------------
        k = random.choice([1, 2, 3])
        inputs_rot = torch.rot90(inputs, k=k, dims=[2, 3])
        cls_rot, segs_rot, attns_rot, attn_pred_rot = wetr(inputs_rot, seg_detach=cfg.train.seg_detach)

        inv_k = (4 - k) % 4
        segs_r_back = torch.rot90(segs_rot, k=inv_k, dims=[2, 3])

        loss_rot_seg = sym_kl_with_confidence(segs, segs_r_back, T=0.7, conf_thr=0.6)

        N = attn_pred.shape[-1]
        Ht = int(round(N ** 0.5))
        Wt = Ht
        if Ht * Wt != N:
            Ht, Wt = get_down_size((inputs.shape[2], inputs.shape[3]), stride=cfg.backbone.stride)

        attn_rot_back = attn_back_from_rot(attn_pred_rot, Ht, Wt, k=k)

        attn_a = torch.logit(attn_pred.clamp(1e-4, 1 - 1e-4))
        attn_b = torch.logit(attn_rot_back.clamp(1e-4, 1 - 1e-4))
        loss_rot_attn = F.mse_loss(attn_a, attn_b)

        loss_rot = loss_rot_seg + loss_rot_attn
        loss_rot = torch.where(
            torch.isnan(loss_rot),
            torch.tensor(0.0, device=loss_rot.device, requires_grad=True),
            loss_rot,
        )
        # ----------------- Joint Rotation Consistency (JRC) -----------------

        refined_pseudo_label = camutils.refine_cams_with_bkg_v2(par, inputs_denorm, cams=cams, cls_labels=cls_labels,
                                                                cfg=cfg,
                                                                img_box=img_box)
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long),
                                        ignore_index=cfg.dataset.ignore_index)

        cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)

        if n_iter <= cfg.train.cam_iters:
            loss = 1.0 * cls_loss + 0.0 * seg_loss
        else:
            loss = 1.0 * cls_loss + 0.1 * seg_loss + 0.1 * loss_rot

        avg_meter.add({'loss': loss.item(), 'cls_loss': cls_loss.item(), 'seg_loss': seg_loss.item(),'loss_rot': loss_rot.item(),'loss_rot_seg': loss_rot_seg.item(),'loss_rot_attn': loss_rot_attn.item(),
                       })

        acc, recall = pyutils.multi_label_accuracy(cls, cls_labels)
        avg_meter.add({'acc': acc, 'recall': recall})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (n_iter + 1) % cfg.train.eval_iters == 0:
            loss = avg_meter.pop('loss')
            acc = avg_meter.pop('acc')
            recall = avg_meter.pop('recall')
            list_meter.add({"loss": loss, "acc": acc, "recall": recall})

            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "wetr_iter_%d.pth" % (n_iter + 1))
            torch.save(wetr.state_dict(), ckpt_name)

    return True


if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8006'
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)


    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, timestamp, cfg.work_dir.ckpt_dir)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir,timestamp, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, timestamp, cfg.work_dir.tb_logger_dir)
    cfg.work_dir.history = os.path.join(cfg.work_dir.dir, timestamp, cfg.work_dir.history)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.history, exist_ok=True)
    ## fix random seed
    setup_seed(1)
    cfg.dist.world_size = 2
    torch.multiprocessing.spawn(train, args=(cfg.dist.world_size, cfg), nprocs=cfg.dist.world_size, join=True)

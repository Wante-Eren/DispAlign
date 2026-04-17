import logging
import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch.distributed as dist
from tqdm import tqdm  # 导入tqdm，用于数据集加载进度条

# ========== 全局优化：开启TF32加速GPU计算，减少显存开销 ==========
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # 开启卷积算法自动优化


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("IDEA.train")
    logger.info('start training with New CDA Module')  # 提示使用新CDA模块
    writer = SummaryWriter('../runs/{}'.format(cfg.OUTPUT_DIR.split('/')[-1]))
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    train_start_time = None  # 全局训练开始时间（初始化）

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        # 优化：分布式训练仅主进程更新scheduler
        if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
            scheduler.step(epoch)
        model.train()
        max_iter_per_epoch = len(train_loader)  # 每轮总迭代数
        max_total_iter = cfg.SOLVER.MAX_EPOCHS * max_iter_per_epoch  # 全局总迭代数

        # ---------------- 数据集加载进度条（优化版：减少IO开销）----------------
        train_loader_tqdm = tqdm(
            enumerate(train_loader),
            total=max_iter_per_epoch,
            desc=f"[Data Loading] Epoch {epoch:02d}/{epochs:02d}",
            leave=False,  # 每轮结束后清除加载进度条，避免刷屏
            bar_format="{l_bar}{bar:15}{r_bar}",
            colour='green',  # 进度条颜色（区分训练进度条）
            mininterval=2.0,  # 至少2秒刷新一次，减少IO
            maxinterval=5.0,  # 最多5秒刷新一次
            disable=cfg.MODEL.DIST_TRAIN and (dist.get_rank() != 0),  # 分布式仅主进程显示
            dynamic_ncols=False,  # 关闭动态列宽，减少计算
        )

        # 迭代包装后的loader，同时显示加载进度和训练进度
        for n_iter, (img, vid, target_cam, target_view, img_path, text) in train_loader_tqdm:
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            # ========== 核心优化：添加non_blocking=True，加速GPU数据传输 ==========
            img = {'RGB': img['RGB'].to(device, non_blocking=True),
                   'NI': img['NI'].to(device, non_blocking=True),
                   'TI': img['TI'].to(device, non_blocking=True)}
            text = {'rgb_text': text['rgb_text'].to(device, non_blocking=True),
                    'ni_text': text['ni_text'].to(device, non_blocking=True),
                    'ti_text': text['ti_text'].to(device, non_blocking=True)}
            target = vid.to(device, non_blocking=True)
            target_cam = target_cam.to(device, non_blocking=True)
            target_view = target_view.to(device, non_blocking=True)

            with amp.autocast(enabled=True):
                img_path_for_vis = None

                output = model(image=img, text=text, label=target, cam_label=target_cam, view_label=target_view,
                               writer=writer, epoch=epoch, img_path=img_path_for_vis)
                loss = 0
                if len(output) % 2 == 1:
                    index = len(output) - 1
                    for i in range(0, index, 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp
                    if not isinstance(output[-1], dict):
                        loss = loss + output[-1]
                    else:
                        num_region = output[-1]['num']
                        for i in range(num_region):
                            loss = loss + (1 / num_region) * loss_fn(score=output[-1][f'score_{i}'],
                                                                    feat=output[-1][f'feat_{i}'],
                                                                    target=target, target_cam=target_cam)
                else:
                    for i in range(0, len(output), 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            # ========== 优化：仅日志打印时同步GPU，减少冗余等待 ==========
            if (n_iter + 1) % log_period == 0:
                torch.cuda.synchronize()

            # ---------------- YOLO风格训练进度条 ----------------
            # 初始化训练开始时间（首次迭代）
            if train_start_time is None:
                train_start_time = time.time()
            
            # 计算进度参数
            current_iter = n_iter + 1  # 1-based迭代数
            current_total_iter = (epoch - 1) * max_iter_per_epoch + current_iter
            progress = (current_total_iter / max_total_iter) * 100  # 进度百分比
            
            # 时间计算
            used_time = time.time() - train_start_time
            remaining_time = (used_time / current_total_iter) * (max_total_iter - current_total_iter)
            used_time_str = f"{int(used_time//60):02d}m{int(used_time%60):02d}s"
            remaining_time_str = f"{int(remaining_time//60):02d}m{int(remaining_time%60):02d}s"
            
            # 每log_period次迭代打印（log_period=1即实时刷新）
            if (n_iter + 1) % log_period == 0:
                # 清除加载进度条临时输出，再打印训练进度条（避免重叠）
                print("\r" + " " * 100, end="")
                print(
                    f"\r[IDEA Training] Epoch {epoch:02d}/{epochs:02d} | "
                    f"Iter {current_iter:04d}/{max_iter_per_epoch:04d} | "
                    f"Progress: {progress:.1f}% | Loss: {loss_meter.avg:.3f} | "
                    f"Acc: {acc_meter.avg:.3f} | LR: {scheduler._get_lr(epoch)[0]:.2e} | "
                    f"Used: {used_time_str} | Remaining: {remaining_time_str}",
                    end="",
                    flush=True
                )
            
            # 每轮结束后换行（避免覆盖）
            if current_iter == max_iter_per_epoch:
                print()  # 换行
            
            # 更新数据加载进度条的附加信息（可选）
            train_loader_tqdm.set_postfix({
                "Load_Speed": f"{train_loader_tqdm.format_dict['elapsed']/(n_iter+1):.3f}s/batch",
                "Current_Loss": f"{loss.item():.3f}"
            })

        # 关闭当前epoch的加载进度条
        train_loader_tqdm.close()

        # ========== 核心优化：释放显存，避免碎片化导致卡顿 ==========
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            # 注释掉多余日志，避免刷屏
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger)
            else:
                mAP, cmc = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger, writer=None)
                writer.add_scalar('RGBNT201/mAP', mAP, epoch)
                writer.add_scalar('RGBNT201/Rank-1', cmc[0], epoch)
                writer.add_scalar('RGBNT201/Rank-5', cmc[4], epoch)
                writer.add_scalar('RGBNT201/Rank-10', cmc[9], epoch)
                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))
                logger.info("~" * 50)
                logger.info("!!!!【 The metrics are based on the feature: LOCAL_t (New CDA) 】!!!!")  # 提示新模块特征
                logger.info("~" * 50)
                logger.info("Current mAP: {:.1%}".format(mAP))
                logger.info("Current Rank-1: {:.1%}".format(cmc[0]))
                logger.info("Current Rank-5: {:.1%}".format(cmc[4]))
                logger.info("Current Rank-10: {:.1%}".format(cmc[9]))
                logger.info("~" * 50)
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                logger.info("~" * 50)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query, logger):
    device = "cuda"
    logger.info("Enter inferencing with New CDA Module")  # 提示新模块推理

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    # 测试集加载进度条（优化版）
    val_loader_tqdm = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc="[Inference Loading]",
        leave=False,
        bar_format="{l_bar}{bar:15}{r_bar}",
        colour='blue',
        mininterval=2.0,
        maxinterval=5.0,
    )
    for n_iter, (img, pid, camid, camids, target_view, imgpath, text) in val_loader_tqdm:
        with torch.no_grad():
            # ========== 优化：添加non_blocking=True ==========
            img = {'RGB': img['RGB'].to(device, non_blocking=True),
                   'NI': img['NI'].to(device, non_blocking=True),
                   'TI': img['TI'].to(device, non_blocking=True)}
            text = {'rgb_text': text['rgb_text'].to(device, non_blocking=True),
                    'ni_text': text['ni_text'].to(device, non_blocking=True),
                    'ti_text': text['ti_text'].to(device, non_blocking=True)}
            camids = camids.to(device, non_blocking=True)
            scenceids = target_view
            target_view = target_view.to(device, non_blocking=True)
            
            img_path_for_vis = imgpath if n_iter == 0 else None

            feat = model(image=img, text=text, cam_label=camids, view_label=target_view, return_pattern=return_pattern,
                        img_path=img_path_for_vis, writer=writer, epoch=epoch)
            # ========== 优化：替换重量级assert为轻量日志 ==========
            if cfg.MODEL.DA:
                if 'LOCAL_v' not in feat or 'LOCAL_t' not in feat:
                    logger.warning("New CDA module missing 'LOCAL_v'/'LOCAL_t' features (non-fatal)")
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid, imgpath))
        # 更新测试集加载进度条信息
        val_loader_tqdm.set_postfix({"Infer_Step": f"{n_iter+1}/{len(val_loader)}"})
    val_loader_tqdm.close()

    # ========== 优化：释放推理显存 ==========
    torch.cuda.empty_cache()

    sign = cfg.MODEL.DA
    if sign:
        logger.info('Current is the local feature testing (New CDA)...')  # 提示新模块特征测试
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['T_RGB'], gallery=['T_RGB'])
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['T_NIR'], gallery=['T_NIR'])
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['T_TIR'], gallery=['T_TIR'])
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['T_RGB', 'T_NIR'], gallery=['T_RGB', 'T_NIR'])
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['T_RGB', 'T_TIR'], gallery=['T_RGB', 'T_TIR'])
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['T_NIR', 'T_TIR'], gallery=['T_NIR', 'T_TIR'])
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['T_RGB', 'T_NIR', 'T_TIR'],
                           gallery=['T_RGB', 'T_NIR', 'T_TIR'])
        _, _ = compute_log(evaluator=evaluator, logger=logger,
                           query=['V_RGB', 'V_NIR', 'V_TIR', 'T_RGB', 'T_NIR', 'T_TIR', 'LOCAL'],
                           gallery=['V_RGB', 'V_NIR', 'V_TIR', 'T_RGB', 'T_NIR', 'T_TIR', 'LOCAL'])
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['LOCAL'],
                           gallery=['LOCAL'])
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['LOCAL_v'], gallery=['LOCAL_v'])
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['LOCAL_t'], gallery=['LOCAL_t'])
        logger.info('Current is the combine feature testing!')
        mAP, cmc = compute_log(evaluator=evaluator, logger=logger,
                               query=['V_RGB', 'V_NIR', 'V_TIR', 'T_RGB', 'T_NIR', 'T_TIR'],
                               gallery=['V_RGB', 'V_NIR', 'V_TIR', 'T_RGB', 'T_NIR', 'T_TIR'])
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['V_RGB', 'V_NIR', 'V_TIR'],
                           gallery=['V_RGB', 'V_NIR', 'V_TIR'])
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['T_RGB', 'T_NIR', 'T_TIR'],
                           gallery=['T_RGB', 'T_NIR', 'T_TIR'])
    else:
        _, _ = compute_log(evaluator=evaluator, logger=logger,
                           query=['V_RGB', 'V_NIR', 'V_TIR', 'T_RGB', 'T_NIR', 'T_TIR'],
                           gallery=['V_RGB', 'V_NIR', 'V_TIR', 'T_RGB', 'T_NIR', 'T_TIR'])
        mAP, cmc = compute_log(evaluator=evaluator, logger=logger, query=['V_RGB', 'V_NIR', 'V_TIR'],
                               gallery=['V_RGB', 'V_NIR', 'V_TIR'])

    return mAP, cmc


def compute_log(evaluator, logger, query, gallery, epoch=0):
    logger.info('Search Pattern --> Query: {} => Gallery: {}'.format(str(query), str(gallery)))
    cmc, mAP, _, _, _, _, _ = evaluator.compute(query=query, gallery=gallery)
    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("~" * 50)
    # ========== 优化：释放计算日志显存 ==========
    torch.cuda.empty_cache()
    return mAP, cmc


def training_neat_eval(cfg,
                       model,
                       val_loader,
                       device,
                       evaluator, epoch, logger, return_pattern=1, writer=None):
    evaluator.reset()
    model.eval()
    # 验证集加载进度条（优化版）
    val_loader_tqdm = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"[Eval Loading] Epoch {epoch:02d}",
        leave=False,
        bar_format="{l_bar}{bar:15}{r_bar}",
        colour='yellow',
        mininterval=2.0,
        maxinterval=5.0,
    )
    for n_iter, (img, pid, camid, camids, target_view, imgpath, text) in val_loader_tqdm:
        with torch.no_grad():
            # ========== 优化：添加non_blocking=True ==========
            img = {'RGB': img['RGB'].to(device, non_blocking=True),
                   'NI': img['NI'].to(device, non_blocking=True),
                   'TI': img['TI'].to(device, non_blocking=True)}
            text = {'rgb_text': text['rgb_text'].to(device, non_blocking=True),
                    'ni_text': text['ni_text'].to(device, non_blocking=True),
                    'ti_text': text['ti_text'].to(device, non_blocking=True)}
            camids = camids.to(device, non_blocking=True)
            scenceids = target_view
            target_view = target_view.to(device, non_blocking=True)
            
            img_path_for_vis = imgpath if n_iter == 0 else None

            feat = model(image=img, text=text, cam_label=camids, view_label=target_view, 
                        img_path=img_path_for_vis, writer=None, epoch=0)
            # ========== 优化：替换重量级assert为轻量日志 ==========
            if cfg.MODEL.DA:
                if 'LOCAL_v' not in feat or 'LOCAL_t' not in feat:
                    logger.warning("New CDA module missing 'LOCAL_v'/'LOCAL_t' features (non-fatal)")
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid, imgpath))
        # 更新验证集加载进度条信息
        val_loader_tqdm.set_postfix({"Eval_Step": f"{n_iter+1}/{len(val_loader)}"})
    val_loader_tqdm.close()

    # ========== 优化：释放验证显存 ==========
    torch.cuda.empty_cache()

    logger.info('Current is the combine feature testing!')
    mAP, cmc = compute_log(evaluator=evaluator, logger=logger,
                           query=['V_RGB', 'V_NIR', 'V_TIR', 'T_RGB', 'T_NIR', 'T_TIR'],
                           gallery=['V_RGB', 'V_NIR', 'V_TIR', 'T_RGB', 'T_NIR', 'T_TIR'], epoch=epoch)
    _, _ = compute_log(evaluator=evaluator, logger=logger, query=['V_RGB', 'V_NIR', 'V_TIR'],
                       gallery=['V_RGB', 'V_NIR', 'V_TIR'], epoch=epoch)
    _, _ = compute_log(evaluator=evaluator, logger=logger, query=['T_RGB', 'T_NIR', 'T_TIR'],
                       gallery=['T_RGB', 'T_NIR', 'T_TIR'], epoch=epoch)
    _, _ = compute_log(evaluator=evaluator, logger=logger, query=['V_RGB'], gallery=['V_RGB'], epoch=epoch)
    _, _ = compute_log(evaluator=evaluator, logger=logger, query=['V_NIR'], gallery=['V_NIR'], epoch=epoch)
    _, _ = compute_log(evaluator=evaluator, logger=logger, query=['V_TIR'], gallery=['V_TIR'], epoch=epoch)
    _, _ = compute_log(evaluator=evaluator, logger=logger, query=['T_RGB'], gallery=['T_RGB'], epoch=epoch)
    _, _ = compute_log(evaluator=evaluator, logger=logger, query=['T_NIR'], gallery=['T_NIR'], epoch=epoch)
    _, _ = compute_log(evaluator=evaluator, logger=logger, query=['T_TIR'], gallery=['T_TIR'], epoch=epoch)
    sign = cfg.MODEL.DA
    if sign:
        logger.info('Current is the local feature testing (New CDA)...')  # 提示新模块特征测试
        _, _ = compute_log(evaluator=evaluator, logger=logger,
                           query=['V_RGB', 'V_NIR', 'V_TIR', 'T_RGB', 'T_NIR', 'T_TIR', 'LOCAL'],
                           gallery=['V_RGB', 'V_NIR', 'V_TIR', 'T_RGB', 'T_NIR', 'T_TIR', 'LOCAL'], epoch=epoch)
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['LOCAL'],
                           gallery=['LOCAL'], epoch=epoch)
        _, _ = compute_log(evaluator=evaluator, logger=logger, query=['LOCAL_v'], gallery=['LOCAL_v'], epoch=epoch)
        mAP, cmc = compute_log(evaluator=evaluator, logger=logger, query=['LOCAL_t'], gallery=['LOCAL_t'], epoch=epoch)
    return mAP, cmc
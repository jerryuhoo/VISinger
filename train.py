import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, DistributedBucketSampler
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from prepare.phone_map import get_vocab_size

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "80000"

    hps = utils.get_hparams()
    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if rank == 0:
        eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=2,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    vocab_size = get_vocab_size()
    net_g = SynthesizerTrn(
        vocab_size,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, (
        phone,
        phone_lengths,
        phone_dur,
        score,
        score_dur,
        pitch,
        slurs,
        spec,
        spec_lengths,
        wave,
        wave_lengths,
    ) in enumerate(train_loader):
        #
        phone, phone_lengths = phone.cuda(rank, non_blocking=True), phone_lengths.cuda(
            rank, non_blocking=True
        )
        phone_dur = phone_dur.cuda(rank, non_blocking=True)
        score = score.cuda(rank, non_blocking=True)
        score_dur = score_dur.cuda(rank, non_blocking=True)
        pitch = pitch.cuda(rank, non_blocking=True)
        slurs = slurs.cuda(rank, non_blocking=True)

        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
            rank, non_blocking=True
        )
        wave, wave_lengths = wave.cuda(rank, non_blocking=True), wave_lengths.cuda(
            rank, non_blocking=True
        )

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                gt_logw,
                pred_logw,
                gt_lf0,
                pred_lf0,
                ctc_loss,
            ) = net_g(
                phone,
                phone_lengths,
                phone_dur,
                score,
                score_dur,
                pitch,
                slurs,
                spec,
                spec_lengths,
            )

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                mse_loss = nn.MSELoss()
                loss_dur = mse_loss(gt_logw, pred_logw)
                loss_pitch = mse_loss(gt_lf0, pred_lf0)
                loss_gen_all = (
                    loss_gen
                    + loss_fm
                    + loss_mel
                    + loss_kl
                    + loss_dur
                    + loss_pitch
                    + ctc_loss
                )
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()

        # for name, param in net_g.named_parameters():
        #     if param.grad is None:
        #         print(name)

        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = [
                    loss_disc,
                    loss_gen,
                    loss_fm,
                    loss_mel,
                    loss_dur,
                    loss_pitch,
                    ctc_loss,
                ]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                # Amor For Tensorboard display
                if loss_mel > 50:
                    loss_mel = 50
                if loss_kl > 5:
                    loss_kl = 5
                if loss_dur > 100:
                    loss_dur = 100
                # if loss_pitch > 100:
                #     loss_pitch = 100

                logger.info([global_step, lr])
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f}"
                )
                logger.info(
                    f"loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}, loss_dur={loss_dur:.3f}, loss_pitch={loss_pitch:.3f}, ctc_loss={ctc_loss:.3f}"
                )

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                        "loss/g/dur": loss_dur,
                        "loss/g/pitch": loss_pitch,
                        "loss/g/ctc": ctc_loss,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, net_d, eval_loader, writer_eval, epoch, logger)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                keep_num = hps.train.keep_n_models
                eval_interval = hps.train.eval_interval
                if global_step / eval_interval >= keep_num:
                    try:
                        os.remove(
                            os.path.join(
                                hps.model_dir,
                                "G_{}.pth".format(
                                    global_step - keep_num * eval_interval
                                ),
                            )
                        )
                        os.remove(
                            os.path.join(
                                hps.model_dir,
                                "D_{}.pth".format(
                                    global_step - keep_num * eval_interval
                                ),
                            )
                        )
                    except OSError:
                        pass

        global_step += 1

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


def evaluate(hps, generator, discriminator, eval_loader, writer_eval, epoch, logger):
    generator.eval()
    loss_disc_avg = 0
    loss_gen_avg = 0
    loss_fm_avg = 0
    loss_mel_avg = 0
    loss_dur_avg = 0
    loss_pitch_avg = 0
    ctc_loss_avg = 0
    loss_kl_avg = 0
    loss_gen_all_avg = 0
    loss_disc_all_avg = 0
    losses_gen_avg = []
    losses_disc_r_avg = []
    losses_disc_g_avg = []
    with torch.no_grad():
        for batch_idx, (
            phone,
            phone_lengths,
            phone_dur,
            score,
            score_dur,
            pitch,
            slurs,
            spec,
            spec_lengths,
            wave,
            wave_lengths,
        ) in enumerate(eval_loader):
            #
            phone, phone_lengths = phone.cuda(0), phone_lengths.cuda(0)
            phone_dur = phone_dur.cuda(0)
            score = score.cuda(0)
            score_dur = score_dur.cuda(0)
            pitch = pitch.cuda(0)
            slurs = slurs.cuda(0)

            spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            wave, wave_lengths = wave.cuda(0), wave_lengths.cuda(0)

            (
                y_hat,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                gt_logw,
                pred_logw,
                gt_lf0,
                pred_lf0,
                ctc_loss,
            ) = generator.module.infer(
                phone,
                phone_lengths,
                phone_dur,
                score,
                score_dur,
                pitch,
                slurs,
                spec,
                spec_lengths,
            )

            y_hat_lengths = x_mask.sum([1, 2]).long() * hps.data.hop_length

            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = discriminator(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc

            with autocast(enabled=hps.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = discriminator(wave, y_hat)
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    mse_loss = nn.MSELoss()
                    loss_dur = mse_loss(gt_logw, pred_logw)
                    loss_pitch = mse_loss(gt_lf0, pred_lf0)
                    loss_gen_all = (
                        loss_gen
                        + loss_fm
                        + loss_mel
                        + loss_kl
                        + loss_dur
                        + loss_pitch
                        + ctc_loss
                    )

                    loss_disc_avg += loss_disc
                    loss_gen_avg += loss_gen
                    loss_fm_avg += loss_fm
                    loss_mel_avg += loss_mel
                    loss_dur_avg += loss_dur
                    loss_pitch_avg += loss_pitch
                    ctc_loss_avg += ctc_loss
                    loss_kl_avg += loss_kl
                    loss_gen_all_avg += loss_gen_all
                    loss_disc_all_avg += loss_disc_all
                    losses_gen_avg = [item + losses_gen for item in losses_gen_avg]
                    losses_disc_r_avg = [
                        item + losses_disc_r for item in losses_disc_r_avg
                    ]
                    losses_disc_g_avg = [
                        item + losses_disc_g for item in losses_disc_g_avg
                    ]

        loss_disc_avg = loss_disc_avg / len(eval_loader)
        loss_gen_avg = loss_gen_avg / len(eval_loader)
        loss_fm_avg = loss_fm_avg / len(eval_loader)
        loss_mel_avg = loss_mel_avg / len(eval_loader)
        loss_dur_avg = loss_dur_avg / len(eval_loader)
        loss_pitch_avg = loss_pitch_avg / len(eval_loader)
        ctc_loss_avg = ctc_loss_avg / len(eval_loader)
        loss_kl_avg = loss_kl_avg / len(eval_loader)
        loss_gen_all_avg = loss_gen_all_avg / len(eval_loader)
        loss_disc_all_avg = loss_disc_all_avg / len(eval_loader)
        losses_gen_avg = [item / len(eval_loader) for item in losses_gen_avg]
        losses_disc_r_avg = [item / len(eval_loader) for item in losses_disc_r_avg]
        losses_disc_g_avg = [item / len(eval_loader) for item in losses_disc_g_avg]

        logger.info("Eval Epoch: {}".format(epoch))
        # Amor For Tensorboard display
        if loss_mel_avg > 50:
            loss_mel_avg = 50
        # if loss_kl > 5:
        #     loss_kl = 5
        if loss_dur_avg > 100:
            loss_dur_avg = 100

        logger.info(
            f"loss_disc={loss_disc_avg:.3f}, loss_gen={loss_gen_avg:.3f}, loss_fm={loss_fm_avg:.3f}"
        )
        logger.info(
            f"loss_mel={loss_mel_avg:.3f}, loss_kl={loss_kl_avg:.3f}, loss_dur={loss_dur_avg:.3f}, loss_pitch={loss_pitch_avg:.3f}, ctc_loss={ctc_loss_avg:.3f}"
        )

        scalar_dict = {
            "loss/g/total": loss_gen_all_avg,
            "loss/d/total": loss_disc_all_avg,
        }
        scalar_dict.update(
            {
                "loss/g/fm": loss_fm_avg,
                "loss/g/mel": loss_mel_avg,
                "loss/g/kl": loss_kl_avg,
                "loss/g/dur": loss_dur_avg,
                "loss/g/pitch": loss_pitch_avg,
                "loss/g/ctc": ctc_loss_avg,
            }
        )

        scalar_dict.update(
            {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen_avg)}
        )
        scalar_dict.update(
            {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r_avg)}
        )
        scalar_dict.update(
            {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g_avg)}
        )

    image_dict = {
        f"gen/mel_{global_step}": utils.plot_spectrogram_to_numpy(
            y_hat_mel[0].cpu().numpy()
        )
    }
    audio_dict = {f"gen/audio_{global_step}": y_hat[0, :, : y_hat_lengths[0]]}
    if global_step == 0:
        image_dict.update(
            {"gt/mel": utils.plot_spectrogram_to_numpy(y_mel[0].cpu().numpy())}
        )
        audio_dict.update({"gt/audio": wave[0, :, : wave_lengths[0]]})

    utils.summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
        scalars=scalar_dict,
    )
    generator.train()


if __name__ == "__main__":
    main()

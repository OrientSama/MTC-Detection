import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import efficientnetv2_m as create_model
from mode_with_fpn import ModelWithFPN
from resnet import resnet50
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
import datetime


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    # 获取当前日期和时间
    now = datetime.datetime.now()
    # 格式化日期和时间
    formatted_date = now.strftime("%Y-%m-%d")
    formatted_time = now.strftime("%H-%M")
    fdt = formatted_date + '-' + formatted_time
    print('Start Tensorboard with "tensorboard --logdir {}", view at http://localhost:6006/'.format(fdt))
    comment = '_{}_{}'.format(args.num_classes, args.epochs)
    tb_writer = SummaryWriter(log_dir="{}".format(fdt), comment=comment)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    multi_label = True if args.num_classes > 1 else False
    train_images_path, train_images_label, val_images_path, val_images_label, train_mask_images_path, val_mask_images_path = read_split_data(
        args.data_path,
        multi_label)

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 512],  # 768-> 384, 1024->512
                "l": [384, 480]}
    num_model = "m"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0], antialias=True),  # antialias=True 打开抗锯齿
                                     transforms.RandomHorizontalFlip()]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model][1], antialias=True),
                                   transforms.CenterCrop(img_size[num_model][1])]),
        "norm": transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              mask_images_path=train_mask_images_path,
                              transform=data_transform["train"],
                              norm=data_transform["norm"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            mask_images_path=val_mask_images_path,
                            transform=data_transform["val"],
                            norm=data_transform["norm"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)
    # model with FPN
    # model = ModelWithFPN(num_classes=args.num_classes).to(device)
    # # ResNet50
    # model = resnet50(num_classes=args.num_classes).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler() if opt.amp else None

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            # load_weights_dict = {k: v for k, v in weights_dict.items()
            #                      if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(weights_dict, strict=False))
            if opt.amp and "scaler" in weights_dict:
                scaler.load_state_dict(weights_dict["scaler"])
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    tags = ["loss", "acc", "learning_rate"]

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                device=device,
                                                epoch=epoch, multilabel=multi_label, warmup=True, scaler=scaler)

        scheduler.step()

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        # validate
        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            val_loss, val_acc = evaluate(model=model, data_loader=val_loader, device=device,
                                         epoch=epoch, multilabel=multi_label)
            tb_writer.add_scalars(tags[0], {'Val': val_loss}, epoch)
            tb_writer.add_scalars(tags[1], {'Val': val_acc}, epoch)

        tb_writer.add_scalars(tags[0], {'Train': train_loss}, epoch)
        tb_writer.add_scalars(tags[1], {'Train': train_acc}, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        torch.save(save_file, f"ASDD_noSOD/EfficientNetv2_1c_{epoch}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1, help="1 or 16 for DOTA1.5")
    parser.add_argument('--epochs', type=int, default=200
                        )
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument("--eval-interval", default=5, type=int, help="validation interval default 10 Epochs")
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default=r"D:\Dataset\ASDD")

    parser.add_argument('--weights', type=str,
                        default=r"",
                        help='initial weights path')
    # parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)

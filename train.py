# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import matplotlib.pyplot as plt

from models import Generator, Discriminator


def show_images(e, x, x_adv, x_fake, save_dir):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1, i].axis("off"), axes[2, i].axis("off")
        axes[0, i].imshow(x[i, 0].cpu().numpy(), cmap="gray")
        axes[0, i].set_title("Normal")
        axes[1, i].imshow(x_adv[i, 0].cpu().numpy(), cmap="gray")
        axes[1, i].set_title("Adv")
        axes[2, i].imshow(x_fake[i, 0].cpu().numpy(), cmap="gray")
        axes[2, i].set_title("APE-GAN")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, "result_{}.png".format(e)))


def main(args):
    lr = args.lr
    epochs = args.epochs
    batch_size = 128
    xi1, xi2 = args.xi1, args.xi2

    check_path = args.checkpoint
    os.makedirs(check_path, exist_ok=True)

    train_data = torch.load("data.tar")
    x_tmp = train_data["normal"][:5]
    x_adv_tmp = train_data["adv"][:5]

    train_data = TensorDataset(train_data["normal"], train_data["adv"])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    G = Generator().cuda()
    D = Discriminator().cuda()

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_bce = nn.BCELoss()
    loss_mse = nn.MSELoss()
    cudnn.benchmark = True

    print_str = "\t".join(["{}"] + ["{:.6f}"] * 2)
    print("\t".join(["{:}"] * 3).format("Epoch", "Gen_Loss", "Dis_Loss"))
    for e in range(epochs):
        G.eval()
        x_fake = G(Variable(x_adv_tmp.cuda())).data
        show_images(e, x_tmp, x_adv_tmp, x_fake, check_path)
        G.train()
        gen_loss, dis_loss, n = 0, 0, 0
        for x, x_adv in tqdm(train_loader, total=len(train_loader), leave=False):
            current_size = x.size(0)
            x, x_adv = Variable(x.cuda()), Variable(x_adv.cuda())
            # Train D
            t_real = Variable(torch.ones(current_size).cuda())
            t_fake = Variable(torch.zeros(current_size).cuda())

            y_real = D(x).squeeze()
            x_fake = G(x_adv)
            y_fake = D(x_fake).squeeze()

            loss_D = loss_bce(y_real, t_real) + loss_bce(y_fake, t_fake)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train G
            for _ in range(2):
                x_fake = G(x_adv)
                y_fake = D(x_fake).squeeze()

                loss_G = xi1 * loss_mse(x_fake, x) + xi2 * loss_bce(y_fake, t_real)
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

            gen_loss += loss_D.data[0] * x.size(0)
            dis_loss += loss_G.data[0] * x.size(0)
            n += x.size(0)
        print(print_str.format(e, gen_loss / n, dis_loss / n))
        torch.save({"generator": G.state_dict(), "discriminator": D.state_dict()},
                   os.path.join(check_path, "{}.tar".format(e + 1)))

    G.eval()
    x_fake = G(Variable(x_adv_tmp.cuda())).data
    show_images(epochs, x_tmp, x_adv_tmp, x_fake, check_path)
    G.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--xi1", type=float, default=0.7)
    parser.add_argument("--xi2", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/test")
    args = parser.parse_args()
    main(args)

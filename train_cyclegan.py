import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from dataset import CTMRIDataset
from models.generator_unet import UNet
from models.patchgan import PatchDiscriminator


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    print("========== CycleGAN 升级版训练开始 ==========")
    device = get_device()
    print("使用设备:", device)

    # 1. 数据集
    dataset = CTMRIDataset(
    ct_dir=r"C:\Users\leili\Desktop\ct_mri_project\data\ct",
    mri_dir=r"C:\Users\leili\Desktop\ct_mri_project\data\mri"
)



    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print("训练样本数量:", len(dataset))

    # 2. 生成器（UNet）
    G_CT2MRI = UNet(1, 1).to(device)
    G_MRI2CT = UNet(1, 1).to(device)

    # 3. 判别器
    D_CT = PatchDiscriminator(1).to(device)
    D_MRI = PatchDiscriminator(1).to(device)

    # 4. 损失
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()   # 用于 cycle 和 id

    # 5. 优化器
    lr = 2e-4
    betas = (0.5, 0.999)

    optimizer_G = Adam(
        list(G_CT2MRI.parameters()) + list(G_MRI2CT.parameters()),
        lr=lr, betas=betas
    )
    optimizer_D_CT = Adam(D_CT.parameters(), lr=lr, betas=betas)
    optimizer_D_MRI = Adam(D_MRI.parameters(), lr=lr, betas=betas)

    # loss 权重
    lambda_cycle = 10.0
    lambda_id = 5.0

    # 输出文件夹
    os.makedirs("weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 创建 CSV 文件（记录每个 epoch 的 loss）
    csv_path = "logs/cyclegan_loss.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "Loss_total",       # Loss_generators + λc * Loss_cycle + λid * Loss_id_total
            "Loss_generators", # ct2mri + mri2ct (GAN)
            "Loss_cycle",      # cycle_ct + cycle_mri
            "Loss_id_total",   # id_ct + id_mri
            "Loss_ct2mri",
            "Loss_mri2ct",
            "Loss_cycle_ct",
            "Loss_cycle_mri",
            "Loss_id_ct",
            "Loss_id_mri",
            "Loss_D_CT",
            "Loss_D_MRI"
        ])

    num_epochs = 100  # 你说想要 50，可以改回 10 / 5

    for epoch in range(num_epochs):
        print(f"\n----- Epoch {epoch+1}/{num_epochs} -----")

        # 这一整轮的累加器
        sum_L_total = sum_L_gen = sum_L_cycle = sum_L_id_total = 0.0
        sum_L_ct2mri = sum_L_mri2ct = 0.0
        sum_L_cycle_ct = sum_L_cycle_mri = 0.0
        sum_L_id_ct = sum_L_id_mri = 0.0
        sum_L_D_CT = sum_L_D_MRI = 0.0

        for i, (real_CT, real_MRI) in enumerate(dataloader):
            real_CT = real_CT.to(device)
            real_MRI = real_MRI.to(device)

            # =============================================
            # 生成器训练
            # =============================================
            optimizer_G.zero_grad()

            fake_MRI = G_CT2MRI(real_CT)
            fake_CT = G_MRI2CT(real_MRI)

            # ------- GAN loss (generator 部分) -------
            pred_fake_MRI = D_MRI(fake_MRI)
            pred_fake_CT = D_CT(fake_CT)

            valid_MRI = torch.ones_like(pred_fake_MRI)
            valid_CT = torch.ones_like(pred_fake_CT)

            loss_ct2mri = criterion_GAN(pred_fake_MRI, valid_MRI)  # Loss_generator_ct2mri
            loss_mri2ct = criterion_GAN(pred_fake_CT, valid_CT)    # Loss_generator_mri2ct
            loss_generators = loss_ct2mri + loss_mri2ct            # Loss_generators

            # ------- Cycle loss -------
            rec_CT = G_MRI2CT(fake_MRI)   # CT -> MRI -> CT
            rec_MRI = G_CT2MRI(fake_CT)   # MRI -> CT -> MRI
            loss_cycle_ct = criterion_L1(rec_CT, real_CT)
            loss_cycle_mri = criterion_L1(rec_MRI, real_MRI)
            loss_cycle = loss_cycle_ct + loss_cycle_mri            # Loss_cycle

            # ------- Identity loss -------
            id_MRI = G_CT2MRI(real_MRI)
            id_CT = G_MRI2CT(real_CT)
            loss_id_mri = criterion_L1(id_MRI, real_MRI)
            loss_id_ct = criterion_L1(id_CT, real_CT)
            loss_id_total = loss_id_ct + loss_id_mri               # Loss_id_total

            # ------- 总生成器 loss（群聊里的 Loss_total）-------
            loss_total = (
                loss_generators
                + lambda_cycle * loss_cycle
                + lambda_id * loss_id_total
            )

            loss_total.backward()
            optimizer_G.step()

            # =============================================
            # 判别器 CT
            # =============================================
            optimizer_D_CT.zero_grad()

            pred_real_CT = D_CT(real_CT)
            loss_D_CT_real = criterion_GAN(pred_real_CT, valid_CT)

            pred_fake_CT = D_CT(fake_CT.detach())
            fake_CT_label = torch.zeros_like(pred_fake_CT)
            loss_D_CT_fake = criterion_GAN(pred_fake_CT, fake_CT_label)

            loss_D_CT = 0.5 * (loss_D_CT_real + loss_D_CT_fake)
            loss_D_CT.backward()
            optimizer_D_CT.step()

            # =============================================
            # 判别器 MRI
            # =============================================
            optimizer_D_MRI.zero_grad()

            pred_real_MRI = D_MRI(real_MRI)
            loss_D_MRI_real = criterion_GAN(pred_real_MRI, valid_MRI)

            pred_fake_MRI = D_MRI(fake_MRI.detach())
            fake_MRI_label = torch.zeros_like(pred_fake_MRI)
            loss_D_MRI_fake = criterion_GAN(pred_fake_MRI, fake_MRI_label)

            loss_D_MRI = 0.5 * (loss_D_MRI_real + loss_D_MRI_fake)
            loss_D_MRI.backward()
            optimizer_D_MRI.step()

            # 统计本 step 的数值，累加到 epoch
            sum_L_total      += loss_total.item()
            sum_L_gen        += loss_generators.item()
            sum_L_cycle      += loss_cycle.item()
            sum_L_id_total   += loss_id_total.item()
            sum_L_ct2mri     += loss_ct2mri.item()
            sum_L_mri2ct     += loss_mri2ct.item()
            sum_L_cycle_ct   += loss_cycle_ct.item()
            sum_L_cycle_mri  += loss_cycle_mri.item()
            sum_L_id_ct      += loss_id_ct.item()
            sum_L_id_mri     += loss_id_mri.item()
            sum_L_D_CT       += loss_D_CT.item()
            sum_L_D_MRI      += loss_D_MRI.item()

            if (i+1) % 50 == 0:
                print(
                    f"Step {i+1}/{len(dataloader)} | "
                    f"L_total: {loss_total.item():.3f} | "
                    f"L_gen: {loss_generators.item():.3f} | "
                    f"L_cycle: {loss_cycle.item():.3f} | "
                    f"L_id: {loss_id_total.item():.3f} | "
                    f"D_CT: {loss_D_CT.item():.3f} | "
                    f"D_MRI: {loss_D_MRI.item():.3f}"
                )

        # ====== 这一轮结束，算平均写进 CSV ======
        n = len(dataloader)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                sum_L_total    / n,
                sum_L_gen      / n,
                sum_L_cycle    / n,
                sum_L_id_total / n,
                sum_L_ct2mri   / n,
                sum_L_mri2ct   / n,
                sum_L_cycle_ct / n,
                sum_L_cycle_mri/ n,
                sum_L_id_ct    / n,
                sum_L_id_mri   / n,
                sum_L_D_CT     / n,
                sum_L_D_MRI    / n, 
            ])

        # 保存模型
        torch.save(G_CT2MRI.state_dict(), f"weights/cyclegan_G_CT2MRI_epoch{epoch+1}.ckpt")
        torch.save(G_MRI2CT.state_dict(), f"weights/cyclegan_G_MRI2CT_epoch{epoch+1}.ckpt")

        print("💾 Loss 已写入 logs/cyclegan_loss.csv")

    print("========== CycleGAN 训练结束 ==========")


if __name__ == "__main__":
    main()

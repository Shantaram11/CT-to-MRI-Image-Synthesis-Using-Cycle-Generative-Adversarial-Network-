import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV （改成你的地址）
csv_path = r"C:\Users\leili\Desktop\ct_mri_project\logs\cyclegan_loss.csv"
df = pd.read_csv(csv_path)

# ================= 图1：Loss_total =================
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["Loss_total"], label="Loss_total")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Total Loss Over Epochs")
plt.grid()
plt.legend()
plt.savefig("loss_total.png", dpi=200)
plt.show()

# ================= 图2：判别器 Loss =================
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["Loss_D_CT"], label="Loss_D_CT")
plt.plot(df["epoch"], df["Loss_D_MRI"], label="Loss_D_MRI")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Discriminator Losses")
plt.grid()
plt.legend()
plt.savefig("loss_discriminator.png", dpi=200)
plt.show()

# ================= 图3：Cycle loss =================
plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["Loss_cycle"], label="Loss_cycle")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Cycle Consistency Loss")
plt.grid()
plt.legend()
plt.savefig("loss_cycle.png", dpi=200)
plt.show()

import os
import torch
import cv2

from dataset import CTMRIDataset
from models.generator_unet import UNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    print("使用设备:", device)

    # 1. 载入数据（用你现在 data 里那套路径）
    dataset = CTMRIDataset(
        ct_dir=r"C:\Users\leili\Desktop\ct_mri_project\data\ct",
        mri_dir=r"C:\Users\leili\Desktop\ct_mri_project\data\mri"
    )
    print("可用样本数:", len(dataset))

    # 2. 建立生成器并载入训练好的权重（记得改成你要的 epoch）
    model = UNet(1, 1).to(device)
    ckpt_path = r"C:\Users\leili\Desktop\ct_mri_project\weights\cyclegan_G_CT2MRI_epoch100.ckpt"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("已载入权重:", ckpt_path)

    # 3. 输出文件夹
    out_dir = r"C:\Users\leili\Desktop\ct_mri_project\results\ct2mri"
    os.makedirs(out_dir, exist_ok=True)

    # 想生成哪几张，就把索引写在这里（后面可以让队友自己改）
    sample_indices = [0, 5, 10, 20, 50, 100]

    with torch.no_grad():
        for idx in sample_indices:
            ct, _ = dataset[idx]   # dataset 返回 (ct, mri)，这里只用 ct
            ct_batch = ct.unsqueeze(0).to(device)  # [1,1,256,256]

            fake_mri = model(ct_batch)[0, 0].cpu().numpy()  # [256,256]

            # 还原到 0-255 并保存成 png
            fake_mri_img = (fake_mri * 255.0).clip(0, 255).astype("uint8")
            out_path = os.path.join(out_dir, f"ct{idx}_to_mri.png")
            cv2.imwrite(out_path, fake_mri_img)
            print("保存生成 MRI:", out_path)

    print("推理完成！")


if __name__ == "__main__":
    main()

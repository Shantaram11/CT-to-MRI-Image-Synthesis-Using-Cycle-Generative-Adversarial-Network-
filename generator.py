

import os
import cv2
import numpy as np
import torch

# ⚠️ Adjust this import depending on your project structure:
# If generator_unet.py is in a "models" folder (models/generator_unet.py), keep this:
from generator_unet import UNet
# If it's in the same folder as this file, use instead:
# from generator_unet import UNet


# ---------------------------------------------------------
# 1. Device selection (same logic as teammate)
# ---------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()

# Keep a global singleton model so we don't reload weights every request
_G_CT2MRI = None
_CURRENT_WEIGHT_PATH = None


# ---------------------------------------------------------
# 2. Load UNet generator with ckpt weights
# ---------------------------------------------------------
def load_ct2mri_generator(weight_path: str):
    """
    Load the UNet CT->MRI generator with given ckpt weights.
    Uses a cached global instance so repeated calls are fast.
    """
    global _G_CT2MRI, _CURRENT_WEIGHT_PATH

    if _G_CT2MRI is not None and _CURRENT_WEIGHT_PATH == weight_path:
        return _G_CT2MRI

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weight file not found: {weight_path}")

    model = UNet(in_channels=1, out_channels=1).to(device)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    _G_CT2MRI = model
    _CURRENT_WEIGHT_PATH = weight_path
    return _G_CT2MRI


# ---------------------------------------------------------
# 3. One-shot CT -> MRI inference
# ---------------------------------------------------------
def generate_mri_from_ct(
    ct_image_path: str,
    output_path: str,
    weight_path: str = "cyclegan_G_CT2MRI_epoch10.ckpt",
    image_size: int = 256,  # kept for API compatibility; actual size is fixed at 256
):
    """
    ct_image_path: path to input CT image (any common format)
    output_path:   where to save generated MRI (PNG/JPG)
    weight_path:   ckpt file path for CycleGAN G_CT2MRI
    image_size:    currently fixed at 256 for this model

    Returns:
        output_path
    """
    if not os.path.exists(ct_image_path):
        raise FileNotFoundError(f"CT image not found: {ct_image_path}")

    # 1) Load model
    G_CT2MRI = load_ct2mri_generator(weight_path)

    # 2) Read CT as grayscale, resize to 256x256 (same as training pipeline)
    ct = cv2.imread(ct_image_path, cv2.IMREAD_GRAYSCALE)
    if ct is None:
        raise ValueError(f"Failed to read CT image (None): {ct_image_path}")

    ct = cv2.resize(ct, (image_size, image_size))  # training used 256x256

    # 3) Normalize to [-1, 1] as in training :contentReference[oaicite:2]{index=2}
    ct = ct.astype(np.float32)
    ct = ct / 127.5 - 1.0  # [0,255] -> [-1,1]

    ct_tensor = torch.from_numpy(ct).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

    # 4) Forward pass
    with torch.no_grad():
        fake_mri = G_CT2MRI(ct_tensor)

    # 5) Denormalize from [-1,1] back to [0,255] uint8 for saving :contentReference[oaicite:3]{index=3}
    fake_mri = fake_mri.squeeze().cpu().numpy()
    fake_mri = (fake_mri + 1.0) * 127.5
    fake_mri = np.clip(fake_mri, 0, 255).astype(np.uint8)

    # 🔥 Resize generated MRI to SAME size as original CT
    orig_ct = cv2.imread(ct_image_path, cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = orig_ct.shape[:2]
    fake_mri_resized = cv2.resize(fake_mri, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    # Save result
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(output_path, fake_mri_resized)

    return output_path


if __name__ == "__main__":
    # Quick local test (adjust paths)
    test_ct = "data/ct/ct8.png"
    out_path = "test_generated_mri.png"
    generate_mri_from_ct(
        ct_image_path=test_ct,
        output_path=out_path,
        weight_path="cyclegan_G_CT2MRI_epoch10.ckpt",
    )
    print(f"Saved generated MRI to {out_path}")

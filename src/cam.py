import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from src.models import SmallResNet, resnet18_ft
from src.config import PROC_DIR

def grad_cam_manual(model, img_tensor, target_layer):
    activations = {}
    gradients   = {}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_inp, grad_out):
        gradients["value"] = grad_out[0].detach()

    layer = dict(model.named_modules())[target_layer]
    layer.register_forward_hook(forward_hook)
    layer.register_backward_hook(backward_hook)

    out = model(img_tensor)
    score = torch.sigmoid(out)[0]
    model.zero_grad()
    score.backward()

    act   = activations["value"][0] # C*h*w
    grad  = gradients["value"][0]   # C*h*w
    weights = grad.mean(dim=(1,2))  # C
    cam = (weights[:, None, None] * act).sum(dim=0).cpu().numpy()  # h*w
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() or 1.0)
    return cam

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True, choices=["small_cnn","resnet18"])
    ap.add_argument("--ckpt",     required=True, help="Path to .ckpt file")
    ap.add_argument("--slice_id", required=True, help="Slice ID (no extension)")
    args = ap.parse_args()

    ModelCls = SmallResNet if args.model == "small_cnn" else resnet18_ft
    model = ModelCls()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()


    target_layer = "layer3" if args.model == "small_cnn" else "layer4"

    img_path = f"{PROC_DIR}/{args.slice_id}.png"
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    tensor = tf(img).unsqueeze(0)  # 1*3*160*160


    cam = grad_cam_manual(model, tensor, target_layer)
    cam_up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = (cam_up * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)

    out_path = f"report/figures/{args.slice_id}_{args.model}_cam.png"
    cv2.imwrite(out_path, overlay)
    print("Saved Grad-CAM to", out_path)

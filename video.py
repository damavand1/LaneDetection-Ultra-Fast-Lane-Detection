import torch, cv2, os
import numpy as np
import scipy.special
import torchvision.transforms as transforms
from model.model import parsingNet
from utils.common import merge_config
from data.constant import culane_row_anchor, tusimple_row_anchor
from torchvision.transforms import ToPILImage

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
        row_anchor = culane_row_anchor
        img_w, img_h = 1640, 590
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
        row_anchor = tusimple_row_anchor
        img_w, img_h = 1280, 720
    else:
        raise NotImplementedError

    # -------- Model --------
    net = parsingNet(
        pretrained=False,
        backbone=cfg.backbone,
        cls_dim=(cfg.griding_num + 1, cls_num_per_lane, 4),
        use_aux=False
    ).to(DEVICE)

    #state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    MODEL_PATH = r"D:/0Kahroba-motors/0 Kahroba VisualCortex/Lane (boundaries)/02 Ultra-Fast-Lane-Detection/Ultra-Fast-Lane-Detection/tusimple_18.pth"

    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    state_dict = checkpoint['model']




    net.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    net.eval()

    # -------- Image transform (دقیقاً مثل demo.py) --------
    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    # -------- Video --------
    cap = cv2.VideoCapture("test.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter("output.avi", fourcc, 30.0, (img_w, img_h))

    col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        raw = frame.copy()
        frame = cv2.resize(frame, (img_w, img_h))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ToPILImage()(img)          # 🔑 این خط حیاتی است
        img = img_transform(img).unsqueeze(0).to(DEVICE)
        

        with torch.no_grad():
            out = net(img)

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)

        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)

        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc

        # -------- Draw lanes --------
        for lane_idx in range(out_j.shape[1]):
            if np.sum(out_j[:, lane_idx] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, lane_idx] > 0:
                        x = int(out_j[k, lane_idx] * col_sample_w * img_w / 800) - 1
                        y = int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        cv2.imshow("UFLD Video", frame)
        vout.write(frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    vout.release()
    cv2.destroyAllWindows()

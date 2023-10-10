import torch

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)

# angle cost
def SIoU_loss(box1, box2, theta=4):
    eps = 1e-7
    cx_pred = (box1[:, 0] + box1[:, 2]) / 2
    cy_pred = (box1[:, 1] + box1[:, 3]) / 2
    cx_gt = (box2[:, 0] + box2[:, 2]) / 2
    cy_gt = (box2[:, 1] + box2[:, 3]) / 2

    dist = ((cx_pred - cx_gt)**2 + (cy_pred - cy_gt)**2) ** 0.5
    ch = torch.max(cy_gt, cy_pred) - torch.min(cy_gt, cy_pred)
    x = ch / (dist + eps)

    angle = 1 - 2*torch.sin(torch.arcsin(x)-torch.pi/4)**2
    # distance cost
    xmin = torch.min(box1[:, 0], box2[:, 0])
    xmax = torch.max(box1[:, 2], box2[:, 2])
    ymin = torch.min(box1[:, 1], box2[:, 1])
    ymax = torch.max(box1[:, 3], box2[:, 3])
    cw = xmax - xmin
    ch = ymax - ymin
    px = ((cx_gt - cx_pred) / (cw+eps))**2
    py = ((cy_gt - cy_pred) / (ch+eps))**2
    gama = 2 - angle
    dis = (1 - torch.exp(-1 * gama * px)) + (1 - torch.exp(-1 * gama * py))

    #shape cost
    w_pred = box1[:, 2] - box1[:, 0]
    h_pred = box1[:, 3] - box1[:, 1]
    w_gt = box2[:, 2] - box2[:, 0]
    h_gt = box2[:, 3] - box2[:, 1]
    ww = torch.abs(w_pred - w_gt) / (torch.max(w_pred, w_gt) + eps)
    wh = torch.abs(h_gt - h_pred) / (torch.max(h_gt, h_pred) + eps)
    omega = (1 - torch.exp(-1 * wh)) ** theta + (1 - torch.exp(-1 * ww)) ** theta

    #IoU loss
    lt = torch.max(box1[..., :2], box2[..., :2])  # [B, rows, 2]
    rb = torch.min(box1[..., 2:], box2[..., 2:])  # [B, rows, 2]

    wh = fp16_clamp(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]
    area1 = (box1[..., 2] - box1[..., 0]) * (
            box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (
            box2[..., 3] - box2[..., 1])
    iou = overlap / (area1 + area2 - overlap)

    SIoU = 1 - iou + (omega + dis) / 2
    return SIoU, iou
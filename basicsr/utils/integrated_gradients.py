import cv2
import torch
import numpy as np
import torchvision
from PIL import Image
from scipy import stats
import matplotlib.pyplot as plt


def cv2_to_pil(img):
    image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    return image

def pil_to_cv2(img):
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return image

def make_pil_grid(pil_image_list):
    sizex, sizey = pil_image_list[0].size
    for img in pil_image_list:
        assert sizex == img.size[0] and sizey == img.size[1], 'check image size'

    target = Image.new('RGB', (sizex * len(pil_image_list), sizey))
    left = 0
    right = sizex
    for i in range(len(pil_image_list)):
        target.paste(pil_image_list[i], (left, 0, right, sizey))
        left += sizex
        right += sizex
    return target

def vis_saliency(map):
    cmap = plt.get_cmap('seismic')
    map_color = (255 * cmap(map * 0.5 + 0.5)).astype(np.uint8)
    return Image.fromarray(map_color)

def vis_saliency_kde(map):
    grad_flat = map.reshape((-1))
    datapoint_y, datapoint_x = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    Y, X = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    pixels = np.vstack([datapoint_x.ravel(), datapoint_y.ravel()])
    kernel = stats.gaussian_kde(pixels, weights=grad_flat)
    Z = np.reshape(kernel(positions).T, map.shape)
    Z = Z / Z.max()
    cmap = plt.get_cmap('seismic')
    map_color = (255 * cmap(Z * 0.5 + 0.5)).astype(np.uint8)
    return Image.fromarray(map_color)


def attr_grad(tensor, h, w, window=8):
    h_x = tensor.size()[2]
    w_x = tensor.size()[3]
    h_grad = torch.pow(tensor[:, :, :h_x - 1, :] - tensor[:, :, 1:, :], 2)
    w_grad = torch.pow(tensor[:, :, :, :w_x - 1] - tensor[:, :, :, 1:], 2)
    grad = torch.pow(h_grad[:, :, :, :-1] + w_grad[:, :, :-1, :], 1 / 2)
    crop = grad[:, :, h: h + window, w: w + window]

    return torch.sum(crop)


def LinearPath(x, fold, gt):
    baseline = np.ones_like(x)
    diff = x - baseline
    l = np.linspace(0, 1, fold).reshape((fold, 1, 1, 1)) 
    image_interpolation = l * diff + baseline

    diffs = np.repeat(np.expand_dims(x - baseline, axis=0), fold, axis=0)
    return image_interpolation.astype(np.float32), diffs.astype(np.float32)
    

def IntegratedGradients(inp, gt, model, save_img_path, h=50, w=50, window_size=16, fold=50, alpha=0.5):

    if not isinstance(inp, np.ndarray):
        inp = inp.detach().cpu().numpy()
    if not isinstance(gt, np.ndarray):
        gt = gt.detach().cpu().numpy()
    if inp.ndim == 4:
        inp = inp[0]
    if gt.ndim == 4:
        gt = gt[0]

    image_interpolation, diffs = LinearPath(inp,fold,gt)
    grad_list = np.zeros_like(image_interpolation)
    result_list = []

    for i in range(fold):
        img_tensor = torch.from_numpy(image_interpolation[i])
        img_tensor.requires_grad_(True)
        
        result = model(img_tensor.unsqueeze(0).cuda())
        target = attr_grad(result, h, w, window_size)
        target.backward()
        grad = img_tensor.grad.cpu().numpy()
        if np.any(np.isnan(grad)):
            grad[np.isnan(grad)] = 0.0
        result = result.detach().cpu()
      
        grad_list[i] = grad * diffs[i]
        result_list.append(result.numpy())

    grad = grad_list.mean(axis=0)
    result = np.asarray(result_list)[-1]
    
    grad_2d = np.abs(grad.sum(axis=0))
    grad_max = grad_2d.max()
    grad = grad_2d / grad_max
    
    inp = cv2.cvtColor(inp.transpose(1,2,0) * 255, cv2.COLOR_RGB2BGR)
    draw_img = inp.copy()
    cv2.rectangle(draw_img, (w, h), (w + window_size, h + window_size), (0, 0, 255), 1)
    position_pil = cv2_to_pil(draw_img)
    saliency_image_abs = vis_saliency(grad)
    saliency_image_kde = vis_saliency_kde(grad)
    blend_abs_and_input = cv2_to_pil(pil_to_cv2(saliency_image_abs) * (1.0 - alpha) + inp * alpha)
    blend_kde_and_input = cv2_to_pil(pil_to_cv2(saliency_image_kde) * (1.0 - alpha) + inp * alpha)
    pil = make_pil_grid(
        [position_pil,
        saliency_image_abs,
        blend_abs_and_input,
        blend_kde_and_input,
        cv2_to_pil(cv2.cvtColor(np.clip(result, a_min=0., a_max=1.).squeeze(0).transpose(1,2,0)*255, cv2.COLOR_RGB2BGR))]
    )
    
    pil.save(save_img_path)
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
from model.train_utils import str2bool, extract_coords, iou, compute_binary_acc, evaluate_model, str2bool
from tqdm import tqdm

def draw_figure(img, heatmap, label_mask=None, heatmap_true=None, threshold=0, coords=None, target=None):
    heatmap = heatmap.copy()
    fig, axes = plt.subplots(4, 1, figsize=(3, 12))
    img = img[:, :, 0]
    img_small = cv2.resize(img, (224, 224))
    axes[0].imshow(img, cmap=plt.cm.Greys_r)
    axes[0].set_title('Original Image')
    axes[1].imshow(heatmap)
    axes[1].set_title('Model Prediction')

    if label_mask is not None:
        heatmap -= threshold
        heatmap[heatmap < 0] = 0
        axes[2].imshow(img_small, cmap=plt.cm.Greys_r)
        axes[2].imshow(heatmap, alpha=0.3, cmap=plt.cm.jet)
        if coords is not None and len(coords) > 0:
            for coord in coords:
                bbox = generate_box(coord)
                axes[2].add_patch(bbox)
        axes[2].set_title('Threshold Prediction')
        if target is not None and -1 not in target:
            rect1, rect2 = generate_box(target, is_pred=False)
            axes[2].add_patch(rect1)
            axes[2].add_patch(rect2)

    if heatmap_true is not None:
        axes[3].imshow(img_small, cmap=plt.cm.Greys_r)
        axes[3].imshow(heatmap_true, alpha=0.3, cmap=plt.cm.jet)
        axes[3].set_title('Generated Heatmap')


    return fig, axes


def generate_box(coord, is_pred=True):
    if is_pred:
        x = coord['x']
        y = coord['y']
        width = coord['width']
        height = coord['height']
        x1 = x - width
        x2 = x + width
        y1 = y - height
        y2 = y + height
        rect = patches.Rectangle((y1, x1), y2 - y1, x2 - x1, linewidth=1, edgecolor='r', facecolor='none')
        return rect
    else:
        coord = coord * 224
        x1, y1, x2, y2 = coord[:4]
        rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')


        x1, y1, x2, y2 = coord[4:]
        rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='b', facecolor='none')

        return rect1, rect2


def evaluate_model(model, test_loader, save_dir, fig_save_dir, device, args):
    model.eval()
    print('Start Evaluation ...')
    file_names = []
    knee_labels = []
    knee_labels_bboxs = []
    knee_pred_bboxs = []
    knee_pred_labels = []
    batch_num = 0
    for img_batch, mask_batch, heatmap_batch, width_batch, height_batch, fnames, targets in tqdm(test_loader):
        if batch_num == 20 and debug:
            break
        batch_num += 1
        img_batch = img_batch.float().to(device)
        with torch.no_grad():
            output = model(img_batch)
            output = output[-1]
            output = output['hm'] if type(output) is dict else output
        output = output.data.cpu().numpy()
        img_batch = img_batch.data.cpu().numpy()
        mask_batch = mask_batch.data.cpu().numpy()
        heatmap_batch = heatmap_batch.data.cpu().numpy()
        file_names.extend(fnames)
        targets = torch.cat(targets, dim=0)
        targets = targets.reshape((8, -1)).transpose(dim0=0, dim1=1)
        # print(targets)
        targets = targets.data.cpu().numpy()
        knee_labels_bboxs.append(targets)
        for img, out, mask, heat, fname, target in zip(img_batch, output, mask_batch, heatmap_batch, fnames, targets):
            # get unprocessed value
            print(fname)
            fname = fname.split('/')[-1]
            img = np.transpose(img, (1, 2, 0))
            heatmap = out[0].copy()
            coords, pred_bbox, pred_label = extract_coords(out, args.threshold)
            if fig_save_dir is not None:
                fig, axes = draw_figure(img, heatmap, mask, heat, args.threshold, coords, target)
                plt.savefig(fig_save_dir + '/{}.png' \
                            .format(fname.replace('.h5', '')))
                plt.close()
            if mask.sum() == 0:
                knee_labels.append([0])
            else:
                knee_labels.append([1])
            knee_pred_labels.append(pred_label)
            knee_pred_bboxs.append(pred_bbox)

    # metrics
    with open(save_dir + '/stats.txt', 'w') as f:

        knee_labels = np.array(knee_labels)
        knee_pred_labels = np.array(knee_pred_labels)
        iou_l, iou_r = iou(knee_labels_bboxs, knee_pred_bboxs, knee_labels)
        iou_total = (iou_l.mean() + iou_r.mean()) / 2
        acc = compute_binary_acc(knee_labels, knee_pred_labels)
        print('IOU', iou_total, 'ACC', acc, file=f)

    # save annotation
    knee_labels_bboxs = np.vstack(knee_labels_bboxs)
    knee_pred_bboxs = np.vstack(knee_pred_bboxs)
    print(knee_pred_bboxs.shape)
    cols = ['pred_bbox_' + str(i) for i in range(8)]
    df = pd.DataFrame(knee_pred_bboxs)
    df.columns = cols
    df['file_names'] = file_names
    if knee_labels.shape[0] != 0:
        df['label'] = knee_labels
    df['pred_label'] = knee_pred_labels

    df.to_csv(save_dir + '/annotation.csv', index=False)
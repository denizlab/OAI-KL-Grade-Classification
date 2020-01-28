import pandas as pd
import numpy as np
def metrics_iou(boxA,boxB):
    '''
    Two numpy array as input
    :param boxA:
    :param boxB:
    :return:
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    # first compute left
    xA = np.maximum(boxA[:, 0], boxB[:, 0])
    yA = np.maximum(boxA[:, 1], boxB[:, 1])
    xB = np.minimum(boxA[:, 2], boxB[:, 2])
    yB = np.minimum(boxA[:, 3], boxB[:, 3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
    boxBArea = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou_l = interArea / (boxAArea + boxBArea - interArea)

    # compute right side
    xA = np.maximum(boxA[:, 4], boxB[:, 4])
    yA = np.maximum(boxA[:, 5], boxB[:, 5])
    xB = np.minimum(boxA[:, 6], boxB[:, 6])
    yB = np.minimum(boxA[:, 7], boxB[:, 7])

    # compute the area of intersection rectangle
    interArea = (xB - xA ) *(yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[:, 6] - boxA[:, 4]) * (boxA[:, 7] - boxA[:, 5])
    boxBArea = (boxB[:, 6] - boxB[:, 4]) * (boxB[:, 7] - boxB[:, 5])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou_r = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value

    return iou_l,iou_r
df = pd.read_csv('./test_output.csv')

boxA = df.iloc[:,:8].values
boxB = df.iloc[:,8:16].values
ioul, iour = metrics_iou(boxA, boxB)
cutoff=0.85
iou = (ioul + iour) / 2
print(iou.shape)
print(iou[iou > cutoff].shape)
print(iou[iou > cutoff].shape[0] / iou.shape[0])
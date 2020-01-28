from model.model import ResNet, ResNetDetection, CombinedLoss, MSELoss
from model.dataloader import *
from model.train_utils import *
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
import time
import torch.optim as optim
import argparse
print('Start training')
parser = argparse.ArgumentParser()
parser.add_argument('-lm', '--load-model', type=str, default=None, dest='load_model')


def main(args):
    model_dir = args.load_model
    test_contents = './bounding_box_oulu/test.csv'
    test_df = pd.read_csv(test_contents)#.sample(n=32).reset_index()
    try:
        test_df.drop(['index'], axis = 1, inplace=True)
    except KeyError:
        pass
    USE_CUDA = torch.cuda.is_available()
    tensor_transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        lambda x: x.float(),
                    ])

    dataset_test = KneeDetectionDataset(test_df, tensor_transform_train, stage='test')

    test_loader = data.DataLoader(dataset_test, batch_size=8)

    # net = ResNet(pretrained=True, dropout=0.2, use_cuda=USE_CUDA)

    if USE_CUDA:
        #net.load_state_dict(torch.load(model_dir))
        net = torch.load(model_dir)
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in net.parameters()])))

    criterion = MSELoss()
    val_loss, all_names, all_labels, all_preds = validate_epoch(net, criterion, test_loader, USE_CUDA)
    iou_l, iou_r = iou(all_labels, all_preds)
    # acc = compute_binary_acc(all_knee_labels, all_knee_preds)
    iou_total = (iou_l.mean() + iou_r.mean()) / 2
    print('IoU:{};'.format(iou_total))
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    df = [all_labels, all_preds]
    df = np.hstack(df)
    df = pd.DataFrame(df)
    print(df.shape)
    df['file_path'] = all_names
    df.to_csv('test_output.csv', index=False)
    print('Val Loss {}'.format(val_loss))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

import os
import sys
import json

import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torchmetrics
from torchsummary import summary

# from model_v2 import MobileNetV2
from model_v3 import mobilenet_v3_small

def main():
    batch_size = 32
    num_classes = 5

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))  # get data root path
    image_path = os.path.join(data_root, "LPN", "data_sequential_img")  # flower data set path
    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    predict_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                           transform=transform)
    predict_num = len(predict_dataset)
    predict_loader = torch.utils.data.DataLoader(predict_dataset,
                                                 batch_size=batch_size, shuffle=True,
                                                 num_workers=nw)
    print(predict_loader)

    # create model
    net = mobilenet_v3_small(num_classes=5)

    model_weight_path = "./MobileNetV3_small_seq.pth"
    net.load_state_dict(torch.load(model_weight_path))
    net.eval()
    # acc = 0.0  # accumulate accurate number / epoch
    test_acc = torchmetrics.Accuracy()
    test_recall = torchmetrics.Recall(average='none', num_classes=num_classes)
    test_precision = torchmetrics.Precision(average='none', num_classes=num_classes)
    test_f1 = torchmetrics.F1Score(average='none', num_classes=num_classes)
    test_auc = torchmetrics.AUROC(average="macro", num_classes=num_classes)
    summary(net, (3, 244, 244))

    with torch.no_grad():
        predict_bar = tqdm(predict_loader, file=sys.stdout)
        for predict_data in predict_bar:
            predict_images, predict_labels = predict_data
            outputs = net(predict_images)
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            # acc += torch.eq(predict_y, predict_labels).sum().item()
            test_acc(predict_y, predict_labels)
            test_auc.update(outputs, predict_labels)
            test_recall(predict_y, predict_labels)
            test_precision(predict_y, predict_labels)
            test_f1(predict_y, predict_labels)

    # predict_accurate = acc / predict_num
    total_acc = test_acc.compute()
    total_recall = test_recall.compute()
    total_precision = test_precision.compute()
    total_auc = test_auc.compute()
    total_f1 = test_f1.compute()

    # print('predict_accuracy: %.4f' %(predict_accurate))

    print("torch metrics acc:", total_acc)
    print("recall of every test dataset class: ", total_recall)
    print("precision of every test dataset class: ", total_precision)
    print("F1-score of every test dataset class: ", total_f1)
    print("auc:", total_auc.item())

    # 清空计算对象
    test_precision.reset()
    test_acc.reset()
    test_recall.reset()
    test_auc.reset()


if __name__ == '__main__':
    main()

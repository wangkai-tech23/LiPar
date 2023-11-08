import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
# from par_LSTM import ParLSTM
from CANet import CANet


def main():
    batch_size = 32
    num_classes = 5
    epochs = 15
    input_size = 9
    hidden_size = 32
    num_layers = 1

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # transform = transforms.Compose([transforms.RandomResizedCrop(16),
    #                                 transforms.RandomHorizontalFlip(),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))  # get data root path
    image_path = os.path.join(data_root, "LPN", "data_sequential_img")  # flower data set path
    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=transform)
    train_num = len(train_dataset)
    print(train_num)

    # # {'normal':0, 'DoS':1, 'Fuzzy':2, 'Gear':3, 'RPM':4}
    # CAN_list = train_dataset.class_to_idx
    # cla_dict = dict((val, key) for key, val in CAN_list.items())
    # # write dict into json file
    # json_str = json.dumps(cla_dict, indent=4)
    #
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=transform)
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # create model
    net = CANet(input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, num_classes=num_classes, init_weights=True)

    # # load pretrain weights （使用迁移学习，加载现成的权重）
    # # download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    # model_weight_path = "./mobilenet_v2.pth"
    # assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    # pre_weights = torch.load(model_weight_path, map_location='cpu')  # 载入后为字典类型
    #
    # # delete classifier weights
    # pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    # # 遍历权重字典，看权重中是否含有对应参数，如果不在则直接保存到pre_dict字典当中
    # missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)  # 载入权重
    #
    # # freeze features weights （冻结特征提取的所有权重）
    # for param in net.features.parameters():
    #     param.requires_grad = False

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    save_path = './CANet_15.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.4f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images)
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.4f  val_accuracy: %.4f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')



if __name__ == '__main__':
    main()

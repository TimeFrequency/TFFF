import logging
import argparse
import torch
import os
import random
import config
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from datasets.aircraft import AircraftDataset
from datasets.bird import BirdsDataset
from datasets.car import CarsDataset
from models.TF_fusion_model import UpScaleResnet50
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def init_seeds(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='TFFF parameters')
    parser.add_argument('--dataset', metavar='DIR', default='aircraft', help='aircraft bird car')
    parser.add_argument('--lr-begin', default=0.00125, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--gamma', default=0.02, type=float, metavar='M',
                        help='gamma')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default:8)')
    parser.add_argument('--langbuda', default=2e-6, type=float, help='mini-batch size (default: 16)')
    parser.add_argument('--num_workers', default=8, type=int,
                        metavar='N', help='num_workers')
    parser.add_argument('--num_channels', default=64, type=int,
                        metavar='N', help='num_channels')
    parser.add_argument('--pre-trained', default=True, type=bool, help='load pre-trained model')
    parser.add_argument('--use-penalty', default=False, type=bool, help='use penalty')
    parser.add_argument('--store-pth', default='./parameters/aircraft/TF', type=str, metavar='checkpoint_path',
                        help='path to save checkpoint')

    args = parser.parse_args()
    return args


args = parse_args()
print(args)
init_seeds(seed=2021)

if args.pre_trained:
    store_root = args.store_pth + "/pretrained"
    if args.use_penalty:
        store_root = store_root + "/use_penalty"
    else:
        store_root = store_root + "/no_penalty"
else:
    store_root = args.store_pth + "/no_pretrained"
    if args.use_penalty:
        store_root = store_root + "/use_penalty"
    else:
        store_root = store_root + "/no_penalty"

# dataset
train_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
    ]
)
if args.dataset == "aircraft":
    train_data = AircraftDataset(config.root_of_aircraft+"train", args.num_channels, transform_448=train_transform, train=True)
    test_data = AircraftDataset(config.root_of_aircraft+"test", args.num_channels, transform_448=test_transform, train=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    #model
    net = UpScaleResnet50(100)

elif args.dataset == "bird":
    train_data = BirdsDataset(config.root_of_aircraft+"train", args.num_channels, transform_448=train_transform, train=True)
    test_data = BirdsDataset(config.root_of_aircraft+"test", args.num_channels, transform_448=test_transform, train=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workes)

    # model
    net = UpScaleResnet50(200)

elif args.dataset == "car":
    train_data = CarsDataset(config.root_of_aircraft+"train", args.num_channels, transform_448=train_transform, train=True)
    test_data = CarsDataset(config.root_of_aircraft+"test", args.num_channels, transform_448=test_transform, train=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workes)

    # model
    net = UpScaleResnet50(196)

# Whether to load the pre-trained model
if args.pre_trained:
    model_pretrain_dict = torch.load(config.pretrain_pth)
    net_dict = net.state_dict()

    pretrained_dict_f = []
    for k, v in model_pretrain_dict.items():
        if "conv" in k and "layer" in k:
            new_k = k[0:14] + "_f" + k[14:]
            single_dict = [new_k, v]
            pretrained_dict_f.append(single_dict)
        if "bn" in k and "layer" in k:
            new_k = k[0:12] + "_f" + k[12:]
            single_dict = [new_k, v]
            pretrained_dict_f.append(single_dict)
        if "downsample" in k and "layer" in k:
            new_k = k[0:19] + "_f" + k[19:]
            single_dict = [new_k, v]
            pretrained_dict_f.append(single_dict)
    pretrained_dict_f = dict(pretrained_dict_f)

    pretrained_dict_f = {k: v for k, v in pretrained_dict_f.items() if k in net_dict}
    pretrained_dict_t = {k: v for k, v in model_pretrain_dict.items() if k in net_dict}

    net_dict.update(pretrained_dict_t)
    net_dict.update(pretrained_dict_f)
    net.load_state_dict(net_dict)

    for param in net.parameters():
        param.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.cuda()
optimizer = torch.optim.SGD(
    net.parameters(), lr=args.lr_begin, momentum=0.9, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


with open(store_root + '/train_log.txt', 'w+') as file:
    file.write('Epoch, lr, Train_Loss, Train_Acc_T, Train_Acc_F, Train_Acc_TF, Test_Acc_T, Test_Acc_F, Test_Acc_TF\n')


# L1 penalty
def L1_penalty(value):
    return torch.abs(value).sum()


slim_params = []
for name, param in net.named_parameters():
    if param.requires_grad and name.endswith('weight') and 'bn2' in name:
        if len(slim_params) % 2 == 0:
            slim_params.append(param)
        else:
            slim_params.append(param)

if __name__ == "__main__":
    min_train_loss = float('inf')
    max_eval_acc = 0
    max_train_acc = 0

    for epoch in range(args.epochs):
        print('\n===== Epoch: {} ====='.format(epoch))
        net.train()  # set model to train mode, enable Batch Normalization and Dropout
        lr_now = optimizer.param_groups[0]['lr']
        train_loss = train_correct_T = train_correct_F = train_correct_TF = train_total = idx = 0

        for batch_idx, (inputs, dcts, targets) in enumerate(tqdm(train_loader, ncols=80)):
            idx = batch_idx

            if inputs.shape[0] < args.batch_size:
                continue

            optimizer.zero_grad()  # Sets the gradients to zero
            inputs, dcts, targets = inputs.cuda(), dcts.cuda(), targets.cuda()

            outputs, alpha_beta = net(inputs, dcts)
            # compute loss
            loss = 0
            if args.use_penalty:
                for output in outputs:
                    loss += criterion(output, targets)  # the loss of terms y_T, y_F and y_TF
                    L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])  # L1 penalty
                    loss += args.langbuda * L1_norm
            else:
                for output in outputs:
                    loss += criterion(output, targets)  # the loss of terms y_T, y_F and y_TF

            loss.backward()
            optimizer.step()

            _, predicted_T = torch.max(outputs[0].data, 1)
            _, predicted_F = torch.max(outputs[1].data, 1)
            _, predicted_TF = torch.max(outputs[2].data, 1)
            train_total += targets.size(0)
            train_correct_T += predicted_T.eq(targets.data).cpu().sum()
            train_correct_F += predicted_F.eq(targets.data).cpu().sum()
            train_correct_TF += predicted_TF.eq(targets.data).cpu().sum()
            train_loss += loss.item()

        scheduler.step()

        train_acc_T = 100.0 * float(train_correct_T) / train_total
        train_acc_F = 100.0 * float(train_correct_F) / train_total
        train_acc_TF = 100.0 * float(train_correct_TF) / train_total
        train_loss = train_loss / (idx + 1)
        print(
            'Train | lr: {:.4f} | Loss: {:.4f} | Acc_T: {:.3f}% ({}/{})'.format(
                lr_now, train_loss, train_acc_T, train_correct_T, train_total
            )
        )
        print(
            'Train | lr: {:.4f} | Loss: {:.4f} | Acc_F: {:.3f}% ({}/{})'.format(
                lr_now, train_loss, train_acc_F, train_correct_F, train_total
            )
        )
        print(
            'Train | lr: {:.4f} | Loss: {:.4f} | Acc_TF: {:.3f}% ({}/{})'.format(
                lr_now, train_loss, train_acc_TF, train_correct_TF, train_total
            )
        )

        # save model
        max_type_acc_train = 999
        # save model with highest acc
        if train_acc_T > max_train_acc:
            max_type_acc_train = 0
            max_train_acc_acc = train_acc_T

        if train_acc_F > max_train_acc:
            max_type_acc_train = 1
            max_train_acc = train_acc_F

        if train_acc_TF > max_train_acc:
            max_type_acc_train = 2
            max_train_acc = train_acc_TF

        if max_type_acc_train == 0:
            torch.save(
                net.state_dict(),
                store_root + "/training_epoch{}_acc_T{}.pth".format(epoch, round(train_acc_T, 3)),
                _use_new_zipfile_serialization=False)

        if max_type_acc_train == 1:
            torch.save(
                net.state_dict(),
                store_root + "/training_epoch{}_acc_F{}.pth".format(epoch, round(train_acc_F, 3)),
                _use_new_zipfile_serialization=False)

        if max_type_acc_train == 2:
            torch.save(
                net.state_dict(),
                store_root + "/training_epoch{}_acc_TF{}.pth".format(epoch, round(train_acc_TF, 3)),
                _use_new_zipfile_serialization=False)
        print("[alfa, beta] = ", alpha_beta)
        print("max_train_acc = ", max_train_acc)

        # Evaluating model with test data every epoch
        if max_train_acc >= 90:
            with torch.no_grad():
                net.eval()  # set model to eval mode, disable Batch Normalization and Dropout
                eval_correct_T = eval_correct_F = eval_correct_TF = eval_total = 0
                for _, (inputs, dcts, targets) in enumerate(tqdm(test_loader, ncols=80)):
                    inputs, dcts, targets = inputs.cuda(), dcts.cuda(), targets.cuda()
                    outputs, alpha_beta = net(inputs, dcts)
                    _, predicted_T = torch.max(outputs[0].data, 1)
                    _, predicted_F = torch.max(outputs[1].data, 1)
                    _, predicted_TF = torch.max(outputs[2].data, 1)
                    eval_total += targets.size(0)
                    eval_correct_T += predicted_T.eq(targets.data).cpu().sum()
                    eval_correct_F += predicted_F.eq(targets.data).cpu().sum()
                    eval_correct_TF += predicted_TF.eq(targets.data).cpu().sum()
                eval_acc_T = 100.0 * float(eval_correct_T) / eval_total
                eval_acc_F = 100.0 * float(eval_correct_F) / eval_total
                eval_acc_TF = 100.0 * float(eval_correct_TF) / eval_total
                print(
                    'test| Acc_T: {:.3f}% ({}/{})'.format(
                        eval_acc_T, eval_correct_T, eval_total
                    )
                )
                print(
                    'test| Acc_F: {:.3f}% ({}/{})'.format(
                        eval_acc_F, eval_correct_F, eval_total
                    )
                )
                print(
                    'test| Acc_TF: {:.3f}% ({}/{})'.format(
                        eval_acc_TF, eval_correct_TF, eval_total
                    )
                )

                # Logging
                with open(store_root + '/train_log.txt', 'a+') as file:
                    file.write(
                        '{}, {:.4f}, {:.4f}, {:.3f}%, {:.3f}%, {:.3f}%, {:.3f}%, {:.3f}%, {:.3f}% \n'.format(
                            epoch, lr_now, train_loss, train_acc_T, train_acc_F, train_acc_TF,
                            eval_acc_T, eval_acc_F, eval_acc_TF
                        )
                    )

                max_type_acc = 999
                # save model with highest acc
                if eval_acc_T > max_eval_acc:
                    max_type_acc = 0
                    max_eval_acc = eval_acc_T

                if eval_acc_F > max_eval_acc:
                    max_type_acc = 1
                    max_eval_acc = eval_acc_F

                if eval_acc_TF > max_eval_acc:
                    max_type_acc = 2
                    max_eval_acc = eval_acc_TF

                if max_type_acc == 0:
                    torch.save(
                        net.state_dict(),
                        store_root + "/testing_epoch{}_acc_T{}.pth".format(epoch, round(eval_acc_T, 3)),
                        _use_new_zipfile_serialization=False)

                if max_type_acc == 1:
                    torch.save(
                        net.state_dict(),
                        store_root + "/testing_epoch{}_acc_F{}.pth".format(epoch, round(eval_acc_F, 3)),
                        _use_new_zipfile_serialization=False)

                if max_type_acc == 2:
                    torch.save(
                        net.state_dict(),
                        store_root + "/testing_epoch{}_acc_TF{}.pth".format(epoch, round(eval_acc_TF, 3)),
                        _use_new_zipfile_serialization=False)
                print("max_test = ", max_eval_acc)

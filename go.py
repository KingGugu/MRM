import os
import tqdm
import torch
import torchvision
import numpy as np
import torch.distributions as dist
from utils import fmax
from models import Model
from datasets import GODataset
from torch_geometric.loader import DataLoader
from utils import check_path, show_args_info, create_run_directory


def save_epoch_weights(checkpoint, epoch, weights_dir):
    filename = f"model_epoch_{epoch}.pth"
    save_path = os.path.join(weights_dir, filename)

    torch.save(checkpoint, save_path)
    print(f"Save Epoch {epoch} weight: {save_path}")


def train_ori(epoch, dataloader, loss_fn, args):
    model.train()
    for data, aug_data in tqdm.tqdm(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        y = torch.from_numpy(np.stack(data.y, axis=0)).to(device)
        loss = loss_fn(model(data).sigmoid(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()


def difficulty_scheduler(epoch, args):
    num_epochs = args.num_epochs
    num_cycles = args.num_cycles
    alpha1_start = args.alpha1_start
    alpha2_start = args.alpha2_start
    alpha1_end = args.alpha1_end
    alpha2_end = args.alpha2_end

    progress = epoch / num_epochs
    effective_progress = (progress * num_cycles) % 1.0
    alpha1 = alpha1_start + effective_progress * (alpha1_end - alpha1_start)
    alpha2 = alpha2_start + effective_progress * (alpha2_end - alpha2_start)
    alpha1 = max(alpha1, 1.01)
    alpha2 = max(alpha2, 1.01)
    beta_distribution = dist.Beta(torch.tensor([alpha1]), torch.tensor([alpha2]))

    return beta_distribution


def train_mrm(epoch, dataloader, loss_fn, args):
    model.train()
    beta_dist = difficulty_scheduler(epoch, args)
    for data, aug_data in tqdm.tqdm(dataloader):
        mix_weight = beta_dist.sample(torch.tensor([len(data)])).to('cuda')
        data = data.to(device)
        aug_data = aug_data.to(device)
        optimizer.zero_grad()
        y = torch.from_numpy(np.stack(data.y, axis=0)).to(device)
        x, x_aug = model(data, aug_data, mix_weight)
        loss = loss_fn(x.sigmoid(), y)
        loss += args.gamma * loss_fn(x_aug.sigmoid(), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()


def test(dataloader):
    model.eval()
    # Iterate over the validation data.

    probs = []
    labels = []
    for data, aug_data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            prob = model(data).sigmoid().detach().cpu().numpy()
            y = np.stack(data.y, axis=0)
        probs.append(prob)
        labels.append(y)
    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)

    return fmax(probs, labels)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='CDConv')
    parser.add_argument('--data-dir', default='./protein-data/go', type=str, metavar='N', help='data root directory')
    parser.add_argument('--level', default='cc', type=str, metavar='N',
                        help='mf: molecular function, bp: biological process, cc: cellular component')
    parser.add_argument('--geometric-radius', default=4.0, type=float, metavar='N', help='initial 3D ball query radius')
    parser.add_argument('--sequential-kernel-size', default=15, type=int, metavar='N', help='1D sequential kernel size')
    parser.add_argument('--kernel-channels', default=[24], type=int, metavar='N', help='kernel channels')
    parser.add_argument('--base-width', default=32, type=float, metavar='N', help='bottleneck width')
    parser.add_argument('--channels', nargs='+', default=[128, 256, 512, 1024], type=int, metavar='N',
                        help='feature channels')
    parser.add_argument('--num-epochs', default=500, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--batch-size', default=24, type=int, metavar='N', help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N', help='learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 5e-4)', dest='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--lr-milestones', nargs='+', default=[300, 400], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--ckpt-path', default='', type=str, help='path where to save checkpoint')

    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument("--model_name", default='CSConv-go', type=str)
    parser.add_argument('--model_idx', default=1, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument('--save_every_epoch', action='store_true', help="save the weight every epoch")

    parser.add_argument('--mrm', action='store_true', help="enable MRM")
    parser.add_argument('--num_cycles', default=4, type=int, help='')
    parser.add_argument('--alpha1_start', default=4, type=int, help='initial value of α1')
    parser.add_argument('--alpha2_start', default=14, type=int, help='initial value of α2')
    parser.add_argument('--alpha1_end', default=14, type=int, help='final value of α1')
    parser.add_argument('--alpha2_end', default=4, type=int, help='final value of α2')
    parser.add_argument('--gamma', default=1.0, type=float, help='Strength of the mixed data Loss')
    parser.add_argument('--base_weight', default='', type=str, help='Weights of the model trained on the original data')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    check_path(args.output_dir)
    args_str = f'{args.model_name}-{args.model_idx}'
    run_dir, log_dir, weights_dir = create_run_directory(args, args.output_dir)
    args.log_file = os.path.join(log_dir, args_str + '.txt')
    args.ckpt_path = weights_dir
    show_args_info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, split='train')
    valid_dataset = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, split='valid')
    test_dataset_30 = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, percent=30, split='test')
    test_dataset_40 = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, percent=40, split='test')
    test_dataset_50 = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, percent=50, split='test')
    test_dataset_70 = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, percent=70, split='test')
    test_dataset_95 = GODataset(root=args.data_dir, random_seed=args.seed, level=args.level, percent=95, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_30 = DataLoader(test_dataset_30, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_40 = DataLoader(test_dataset_40, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_50 = DataLoader(test_dataset_50, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_70 = DataLoader(test_dataset_70, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader_95 = DataLoader(test_dataset_95, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = Model(geometric_radii=[2 * args.geometric_radius, 3 * args.geometric_radius, 4 * args.geometric_radius,
                                   5 * args.geometric_radius],
                  sequential_kernel_size=args.sequential_kernel_size,
                  kernel_channels=args.kernel_channels, channels=args.channels, base_width=args.base_width,
                  num_classes=train_dataset.num_classes).to(device)
    if args.mrm:
        model.load_state_dict(torch.load(args.base_weight))
        print("Enable MRM, loading base weight.")
        with open(args.log_file, 'a') as f:
            f.write(f"Enable MRM, loading base weight.\n")
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum)
    loss_fn = torch.nn.BCELoss(weight=torch.as_tensor(train_dataset.weights).to(device))

    # learning rate scheduler
    lr_weights = []
    for i, milestone in enumerate(args.lr_milestones):
        if i == 0:
            lr_weights += [np.power(args.lr_gamma, i)] * milestone
        else:
            lr_weights += [np.power(args.lr_gamma, i)] * (milestone - args.lr_milestones[i - 1])
    if args.lr_milestones[-1] < args.num_epochs:
        lr_weights += [np.power(args.lr_gamma, len(args.lr_milestones))] * (
                args.num_epochs + 1 - args.lr_milestones[-1])
    lambda_lr = lambda epoch: lr_weights[epoch]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    best_valid = best_test_30 = best_test_40 = best_test_50 = best_test_70 = best_test_95 = best_30 = best_40 = best_50 = best_70 = best_95 = 0.0
    best_epoch = 0
    for epoch in range(args.num_epochs):
        if args.mrm:
            train_mrm(epoch, train_loader, loss_fn, args)
        else:
            train_ori(epoch, train_loader, loss_fn, args)
        lr_scheduler.step()
        valid_fmax = test(valid_loader)
        test_30 = test(test_loader_30)
        test_40 = test(test_loader_40)
        test_50 = test(test_loader_50)
        test_70 = test(test_loader_70)
        test_95 = test(test_loader_95)
        print(
            f'Epoch: {epoch + 1:03d}, Validation: {valid_fmax:.4f}, Test: {test_30:.4f} {test_40:.4f} {test_50:.4f} {test_70:.4f} {test_95:.4f}')
        with open(args.log_file, 'a') as f:
            f.write(
                f'Epoch: {epoch + 1:03d}, Validation: {valid_fmax:.4f}, Test: {test_30:.4f} {test_40:.4f} {test_50:.4f} {test_70:.4f} {test_95:.4f} \n')
        if args.save_every_epoch:
            save_epoch_weights(model.state_dict(), epoch + 1, weights_dir)

        if valid_fmax >= best_valid:
            best_valid = valid_fmax
            best_30 = test_30
            best_40 = test_40
            best_50 = test_50
            best_70 = test_70
            best_95 = test_95
            best_epoch = epoch
            checkpoint = model.state_dict()

        best_test_30 = max(test_30, best_test_30)
        best_test_40 = max(test_40, best_test_40)
        best_test_50 = max(test_50, best_test_50)
        best_test_70 = max(test_70, best_test_70)
        best_test_95 = max(test_95, best_test_95)
    print(
        f'\nBest Epoch: {best_epoch + 1:03d}, Validation: {best_valid:.4f}, Test: {best_30:.4f} {best_40:.4f} {best_50:.4f} {best_70:.4f} {best_95:.4f} \n')
    print(
        f'Best Performance: Test: {best_test_30:.4f} {best_test_40:.4f} {best_test_50:.4f} {best_test_70:.4f} {best_test_95:.4f} \n')
    with open(args.log_file, 'a') as f:
        f.write(
            f'\nBest Epoch: {best_epoch + 1:03d}, Validation: {best_valid:.4f}, Test: {best_30:.4f} {best_40:.4f} {best_50:.4f} {best_70:.4f} {best_95:.4f} \n')
        f.write(
            f'Best Performance: Test: {best_test_30:.4f} {best_test_40:.4f} {best_test_50:.4f} {best_test_70:.4f} {best_test_95:.4f} \n')

    save_epoch_weights(checkpoint, best_epoch + 1, weights_dir)

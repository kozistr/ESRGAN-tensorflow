import argparse

args_list = []
parser = argparse.ArgumentParser()


def add_arg_group(name: str):
    """
    :param name: A str. Argument group.
    :return: An list. Arguments.
    """
    arg = parser.add_argument_group(name)
    args_list.append(arg)
    return arg


def get_config():
    cfg, un_parsed = parser.parse_known_args()
    return cfg, un_parsed


# Network
network_arg = add_arg_group('Network')
network_arg.add_argument('--model', type=str, default="esr", choices=["esr"])
network_arg.add_argument('--norm_type', type=str, default="bn", choices=["none", "bn", "in"])
network_arg.add_argument('--n_feats', type=int, default=64)
network_arg.add_argument('--n_blocks', type=int, default=23)
network_arg.add_argument('--n_channels', type=int, default=3)
network_arg.add_argument('--n_group_channels', type=int, default=32)
network_arg.add_argument('--n_groups', type=int, default=1)
network_arg.add_argument('--scale', type=int, default=4)
network_arg.add_argument('--activation', type=str, default="leaky_relu", choices=["relu", "leaky_relu", "prelu"])

# DataSet
data_arg = add_arg_group('DataSet')
data_arg.add_argument('--dataset', type=str, default="D:\\DataSet\\SR\\DIV2K\\")
data_arg.add_argument('--lr_img_size', type=int, default=96, choices=[96, 128])
data_arg.add_argument('--hr_img_size', type=int, default=192, choices=[192, 256])

# Train/Test hyper-parameters
train_arg = add_arg_group('Training')
train_arg.add_argument('--batch_size', type=int, default=16)
train_arg.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
train_arg.add_argument('--d_beta1', type=float, default=.9)
train_arg.add_argument('--g_beta1', type=float, default=.9)
train_arg.add_argument('--d_weight_decay', type=float, default=0.)
train_arg.add_argument('--g_weight_decay', type=float, default=0.)
train_arg.add_argument('--d_lr', type=float, default=1e-4)
train_arg.add_argument('--g_lr', type=float, default=1e-4)
train_arg.add_argument('--lr_decay_steps', type=list, default=[int(5e4), int(1e5), int(2e5), int(3e5)])
train_arg.add_argument('--lr_decay_ratio', type=float, default=.5)
train_arg.add_argument('--pixel_criterion', type=str, default="l1", choices=["l1", "l2"])
train_arg.add_argument('--pixel_weight', type=float, default=1e-2)
train_arg.add_argument('--feature_criterion', type=str, default="l1", choices=["l1", "l2"])
train_arg.add_argument('--feature_weight', type=float, default=1.)
train_arg.add_argument('--gan_weight', type=float, default=5e-3)
train_arg.add_argument('--max_steps', type=int, default=int(5e5))
train_arg.add_argument('--log_steps', type=int, default=int(5e3))
train_arg.add_argument('--checkpoint_path', type=str, default="./model/")

# Misc
misc_arg = add_arg_group('Misc')
misc_arg.add_argument('--device', type=str, default='gpu')
misc_arg.add_argument('--n_threads', type=int, default=8)
misc_arg.add_argument('--seed', type=int, default=13371337)
misc_arg.add_argument('--verbose', type=bool, default=True)

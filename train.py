
import os
import argparse
import random
import logging
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

from utils import __balance_val_split, __split_of_train_sequence, __log_class_statistics
from datasets.Lsp_dataset import LSP_Dataset
from spoter.spoter_model import SPOTER
from spoter.utils import train_epoch, evaluate
from spoter.gaussian_noise import GaussianNoise

import wandb

CONFIG_FILENAME = "config.json"
PROJECT_WANDB = "Spoter-as-orignal"
ENTITY = "joenatan30"

def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="lsa_64_spoter",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=50, help="Number of classes to be recognized by the model")
    parser.add_argument("--hidden_dim", type=int, default=108,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")

    # Data
    parser.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")
    parser.add_argument("--experimental_train_split", type=float, default=None,
                        help="Determines how big a portion of the training set should be employed (intended for the "
                             "gradually enlarging training set experiment from the paper)")

    parser.add_argument("--validation_set", type=str, choices=["from-file", "split-from-train", "none"],
                        default="from-file", help="Type of validation set construction. See README for further rederence")
    parser.add_argument("--validation_set_size", type=float,
                        help="Proportion of the training set to be split as validation set, if 'validation_size' is set"
                             " to 'split-from-train'")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument("--log_freq", type=int, default=1,
                        help="Log frequency (frequency of printing all the training info)")

    # Checkpointing
    parser.add_argument("--save_checkpoints", type=bool, default=True,
                        help="Determines whether to save weights checkpoints")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=int, default=0.1, help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="Patience for the ReduceLROnPlateau scheduler")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=int, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=int, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Visualization
    parser.add_argument("--plot_stats", type=bool, default=True,
                        help="Determines whether continuous statistics should be plotted at the end")
    parser.add_argument("--plot_lr", type=bool, default=True,
                        help="Determines whether the LR should be plotted at the end")

    parser.add_argument("--device", type=int, default=0,
                    help="Determines which Nvidia device will use (just one number)")
    # To continue training the data
    parser.add_argument("--continue_training", type=str, default="",help="path to retrieve the model to continue training")

    return parser

def lr_lambda(current_step, optim):
    if current_step <= 50:
        lr_rate = current_step/50000  # Función lineal
    else:
        lr_rate = (0.00005/current_step) ** 0.5  # Función de raíz cuadrada inversa

    print(f'[{current_step}], Lr_rate: {lr_rate}')
    optim.param_groups[0]['lr'] = lr_rate

    return optim

def train(args):

    # MARK: TRAINING PREPARATION AND MODULES

    # Initialize all the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Set the output format to print into the console and save into LOG file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + ".log")
        ]
    )

    run = wandb.init(project=PROJECT_WANDB, entity=ENTITY, config=args, name=args.experiment_name, job_type="model-training")
    config = wandb.config
    wandb.watch_called = False

    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")

    # Construct the model
    slrt_model = SPOTER(num_classes=args.num_classes, hidden_dim=args.hidden_dim)
    slrt_model.train(True)
    slrt_model.to(device)


    # Construct the other modules
    cel_criterion = nn.CrossEntropyLoss()#label_smoothing=0.1)
    sgd_optimizer = optim.SGD(slrt_model.parameters(), lr=args.lr)
    epoch_start = 0

    if args.continue_training:
        checkpoint = torch.load(args.continue_training)
        slrt_model.load_state_dict(checkpoint['model_state_dict'])
        sgd_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(sgd_optimizer, factor=args.scheduler_factor, patience=args.scheduler_patience)
    #scheduler = optim.lr_scheduler.LambdaLR(sgd_optimizer, lr_lambda=lr_lambda)

    # Ensure that the path for checkpointing and for images both exist
    Path("out-checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)

    # MARK: DATA
    artifact_name = config['experiment_name']
    print("artifact_name : ", artifact_name)    
    model_artifact = wandb.Artifact(artifact_name, type='model')

    print("#"*50)
    print("#"*30)
    print("#"*10)
    print("Num Trainable Params: ", sum(p.numel() for p in slrt_model.parameters() if p.requires_grad))
    print("#"*10)
    print("#"*30)
    print("#"*50)

    # Training set
    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    train_set = LSP_Dataset(args.training_set_path, transform=transform, have_aumentation=True, keypoints_model='mediapipe')

    # Validation set
    if args.validation_set == "from-file":
        val_set = LSP_Dataset(args.validation_set_path, keypoints_model='mediapipe', have_aumentation=False)
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    elif args.validation_set == "split-from-train":
        train_set, val_set = __balance_val_split(train_set, 0.2)

        val_set.transform = None
        val_set.augmentations = False
        val_loader = DataLoader(val_set, shuffle=True, generator=g)

    else:
        val_loader = None

    # Testing set
    if args.testing_set_path:
        eval_set = LSP_Dataset(args.testing_set_path)
        eval_loader = DataLoader(eval_set, shuffle=True, generator=g)

    else:
        eval_loader = None

    # Final training set refinements
    if args.experimental_train_split:
        train_set = __split_of_train_sequence(train_set, args.experimental_train_split)

    train_loader = DataLoader(train_set, shuffle=True, generator=g)


    # MARK: TRAINING
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs, val_accs_top5 = [], [], [], []
    lr_progress = []
    top_train_acc, top_val_acc = 0, 0
    checkpoint_index = 0

    if args.experimental_train_split:
        print("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")
        logging.info("Starting " + args.experiment_name + "_" + str(args.experimental_train_split).replace(".", "") + "...\n\n")

    else:
        print("Starting " + args.experiment_name + "...\n\n")
        logging.info("Starting " + args.experiment_name + "...\n\n")

    for epoch in range(epoch_start, args.epochs):

        sgd_optimizer = lr_lambda(epoch, sgd_optimizer)

        train_loss, _, _, train_acc = train_epoch(slrt_model, train_loader, cel_criterion, sgd_optimizer, device)
        losses.append(train_loss.item())
        train_accs.append(train_acc)

        if val_loader:
            slrt_model.train(False)
            val_loss, _, _, val_acc, val_acc_top5 = evaluate(slrt_model, val_loader, cel_criterion, device)
            slrt_model.train(True)
            val_accs.append(val_acc)
            val_accs_top5.append(val_acc_top5)
            wandb.log({
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_top5_acc': val_acc_top5,
                'val_loss':val_loss,
                'epoch': epoch
            })

        # Save checkpoints if they are best in the current subset
        if args.save_checkpoints:
            if train_acc > top_train_acc:
                top_train_acc = train_acc
                val_acc_top5
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': slrt_model.state_dict(),
                    'optimizer_state_dict': sgd_optimizer.state_dict(),
                    'loss': train_loss
                }, "out-checkpoints/" + args.experiment_name + "/checkpoint_t_" + str(checkpoint_index) + ".pth")
                
            if val_acc > top_val_acc:
                top_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': slrt_model.state_dict(),
                    'optimizer_state_dict': sgd_optimizer.state_dict(),
                    'loss': train_loss
                }, "out-checkpoints/" + args.experiment_name + "/checkpoint_v_" + str(checkpoint_index) + ".pth")

        if epoch % args.log_freq == 0:
            print("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item()) + " acc: " + str(train_acc))
            logging.info("[" + str(epoch + 1) + "] TRAIN  loss: " + str(train_loss.item()) + " acc: " + str(train_acc))

            if val_loader:
                print("[" + str(epoch + 1) + "] VALIDATION  loss: " + str(val_loss.item()) + "acc: " + str(val_acc) + " top-5(acc): " + str(val_acc_top5))
                logging.info("[" + str(epoch + 1) + "] VALIDATION  loss: " + str(val_loss.item()) + "acc: " + str(val_acc) + " top-5(acc): " + str(val_acc_top5))

            print("")
            logging.info("")

        # Reset the top accuracies on static subsets
        if epoch % 10 == 0:
            top_train_acc, top_val_acc, val_acc_top5 = 0, 0, 0
            checkpoint_index += 1

        lr_progress.append(sgd_optimizer.param_groups[0]["lr"])

    # MARK: TESTING

    print("\nTesting checkpointed models starting...\n")
    logging.info("\nTesting checkpointed models starting...\n")

    top_result, top_result_name = 0, ""

    if eval_loader:
        for i in range(checkpoint_index):
            for checkpoint_id in ["t", "v"]:
                # tested_model = VisionTransformer(dim=2, mlp_dim=108, num_classes=100, depth=12, heads=8)
                tested_model = torch.load("out-checkpoints/" + args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i) + ".pth")
                tested_model.train(False)
                _, _, eval_acc = evaluate(tested_model, eval_loader, device, print_stats=True)

                if eval_acc > top_result:
                    top_result = eval_acc
                    top_result_name = args.experiment_name + "/checkpoint_" + checkpoint_id + "_" + str(i)

                print("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))
                logging.info("checkpoint_" + checkpoint_id + "_" + str(i) + "  ->  " + str(eval_acc))

        print("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")
        logging.info("\nThe top result was recorded at " + str(top_result) + " testing accuracy. The best checkpoint is " + top_result_name + ".")


    # PLOT 0: Performance (loss, accuracies) chart plotting
    if args.plot_stats:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses) + 1), losses, c="#D64436", label="Training loss")
        ax.plot(range(1, len(train_accs) + 1), train_accs, c="#00B09B", label="Training accuracy")

        if val_loader:
            ax.plot(range(1, len(val_accs) + 1), val_accs, c="#E0A938", label="Validation accuracy")
            ax.plot(range(1, len(val_accs_top5) + 1), val_accs_top5, c="#F2B09A", label="val_Top5_acc")

        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.set(xlabel="Epoch", ylabel="Accuracy / Loss", title="")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, fancybox=True, shadow=True, fontsize="xx-small")
        ax.grid()
        fig.savefig("out-img/" + args.experiment_name + "_loss.png")

    # PLOT 1: Learning rate progress
    if args.plot_lr:
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1, len(lr_progress) + 1), lr_progress, label="LR")
        ax1.set(xlabel="Epoch", ylabel="LR", title="")
        ax1.grid()

        fig1.savefig("out-img/" + args.experiment_name + "_lr.png")

    print("\nAny desired statistics have been plotted.\nThe experiment is finished.")
    logging.info("\nAny desired statistics have been plotted.\nThe experiment is finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    train(args)

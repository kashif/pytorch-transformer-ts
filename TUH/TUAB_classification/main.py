
import tempfile
import numpy as np 
import pandas
#
import mne
mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted

from sklearn.metrics import accuracy_score, balanced_accuracy_score
#
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.tuh import TUH, TUHAbnormal  # noqa F811

import models

#
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
import torch.nn.functional as F


#
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

@hydra.main(config_path="configs", config_name="main-args")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Device definition
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:",device) 

    # define random seeds
    torch.manual_seed(cfg.args.seed)

    from prepare_data import prepare_tuab
    dataset_tuab = prepare_tuab(cfg.args.TUH_PATH)

    print(dataset_tuab.description)

    # We can finally generate compute windows. The resulting dataset is now ready
    # we will create compute windows. We specify a
    # mapping from genders 'M' and 'F' to integers, since this is required for
    # decoding.

    tuab_windows = create_fixed_length_windows(
        dataset_tuab,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=750,
        window_stride_samples=750,
        drop_last_window=False,
        mapping={'M': 0, 'F': 1, False: 0, True: 1 },  # map non-digit targets
    )
    # store the number of windows required for loading later on
    tuab_windows.set_description({
        "n_windows": [len(d) for d in tuab_windows.datasets]})

    ###############################################################################
    # Iterating through the dataset gives x as ndarray(n_channels x 1000), y as
    # [age, gender], and ind. Let's look at the last example again.
    # print(tuh_windows.description)
    x, y, ind = tuab_windows[-1]
    print('x:', x.shape)
    print('y:', y)
    print('ind:', ind)

    # if cfg.args.split_mode == 'train':
    #     # split by train val
    tuab_splits = tuab_windows.split("train") # splits the datsets to train/val
    

    ###############################################################################
    # We give the dataset to a pytorch DataLoader, such that it can be used for
    # model training.
    dl_train = DataLoader(
        # dataset=tuh_splits[str(splits[3+cfg.args.train_grp])],
        dataset=tuab_splits["True"],
        batch_size=cfg.args.batch_size,
        num_workers=cfg.args.num_workers,
        drop_last=True,
    )
    dl_eval = DataLoader(
        # dataset=tuh_splits[str(splits[cfg.args.val_grp])],
        dataset=tuab_splits["False"],
        batch_size=128,
        num_workers=cfg.args.num_workers,
    )

    # dl_test = DataLoader(
    #     # dataset=tuh_splits[str(splits[cfg.args.val_grp])],
    #     dataset=tuh_windows_test,
    #     batch_size=128,
    #     num_workers=cfg.args.num_workers,
    # )

    ###############################################################################
    # Iterating through the DataLoader gives batch_X as tensor(4 x n_channels x
    # 1000), batch_y as [tensor([4 x age of subject]), tensor([4 x gender of
    # subject])], and batch_ind. We will iterate to the end to look at the last example
    # again.
    for batch_X, batch_y, batch_ind in dl_train:
        break
    print('batch_X:', batch_X.shape)
    print('batch_y:', batch_y)
    print('batch_ind:', len(batch_ind))

    with open_dict(cfg):
        cfg.data.in_chans = batch_X.shape[1]
        cfg.data.n_classes = 2
        cfg.data.input_window_samples=batch_X.shape[2]

    ## training 
    model = models.get_model(cfg)
    print("Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.args.lr, weight_decay=cfg.args.weight_decay)

    #
    train_acc = []
    val_acc = []
    # test_acc = []

    for ii in range(cfg.args.epochs): 
        losses = []
        for batch_X, batch_y, batch_ind in dl_train:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            # batch_y = [y.to(device) for y in batch_y]
            pred = model(batch_X)
            # pred = pred[:,:,0].squeeze()
            # print(pred, batch_y)
            loss = F.cross_entropy(pred, batch_y)
            losses.append(loss.cpu().detach().numpy())

            # Back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_acc.append(validatin(dl_train, model, device))
        val_acc.append(validatin(dl_eval, model, device))
        # test_acc.append(validatin(dl_test, model, device))

        print ('epoch:',ii, 'loss:', np.array(losses).mean(), '| train accuracy:', train_acc[-1] , '| val accuracy:', val_acc[-1])#, '| test accuracy:', test_acc[-1])

    # save model
    torch.save(model.state_dict(), 'd4.pth',_use_new_zipfile_serialization=False)

    #load model 
    model.load_state_dict(torch.load('d4.pth'))

    #Save the size and accuracy
    idx = torch.argmin(torch.tensor(val_acc)[:,1]) # based on the loss [1] or acc [0]
    df = pandas.DataFrame(data={'seed': [cfg.args.seed], 'idx': [idx], 'Acc-train': [train_acc[idx][0]],'Acc-val': [val_acc[idx][0]] })#,'Acc-test': [test_acc[idx][0]], 'Size': [cfg.args.n_sub_train]})
    print(df)
    df.to_csv('./accuracy_runs.csv', sep=',', mode='a', header=False)


def validatin(dl_eval, model, device):
    # validatin
    y_pred = []
    y_true = [] 
    losses = []
    for batch_X, batch_y, batch_ind in dl_eval:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        # batch_y = [y.to(device) for y in batch_y]
        model.eval()
        pred = model(batch_X)
        y_pred.extend(torch.argmax(pred,dim=1).cpu().detach().numpy())
        y_true.extend(batch_y.cpu().numpy())
        # cal loss 
        loss = F.cross_entropy(pred, batch_y)
        losses.append(loss.cpu().detach().numpy())
    # print(y_true,'\n', y_pred)
    # print('accuracy_score:', accuracy_score(np.array(y_pred), np.array(y_true)))
    # print(np.array(losses).mean())
    return balanced_accuracy_score(np.array(y_pred),np.array(y_true)), np.array(losses).mean()


if __name__ == '__main__':
    main()

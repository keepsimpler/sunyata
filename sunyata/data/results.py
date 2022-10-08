import pandas as pd

def from_csv_to_dataframe(hparams: dict, path: str):

    df_hparams = pd.DataFrame.from_dict([hparams])

    df = pd.read_csv(path)

    train_metrics = ['train_loss', 'train_accuracy', 'step', 'epoch']
    val_metrics = ['val_loss', 'val_accuracy', 'epoch']

    df_train_metrics = df.loc[:, train_metrics]
    df_train_metrics = df_train_metrics.dropna()

    df_train_metrics_by_epoch = df_train_metrics.groupby(['epoch'])[['train_loss', 'train_accuracy']].mean()

    df_val_metrics = df.loc[:, val_metrics]
    df_val_metrics = df_val_metrics.dropna()
    df_val_metrics = df_val_metrics.set_index('epoch')

    df_metrics = pd.concat([df_train_metrics_by_epoch, df_val_metrics], axis=1, join='inner')

    df_metrics['epoch'] = df_metrics.index

    df_metrics.index = [0] * len(df_metrics)

    df_all = pd.concat([df_metrics, df_hparams], axis=1, join='outer')

    return df_all
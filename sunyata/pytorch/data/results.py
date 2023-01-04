import os
import yaml
import pandas as pd

def list_hparams_yaml_filenames(path: str, suffix: str):
    '''
    list yaml filenames with a suffix in a path, such as: convmixer.yaml
    '''
    full_filenames = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            full_filenames.append(os.path.join(dirname, filename))

    hparams_yaml_filenames = [filename for filename in full_filenames if filename.endswith(f'.{suffix}.yaml')]
    return hparams_yaml_filenames


def from_csv_to_dataframe(path: str):
    df = pd.read_csv(path)

    train_metrics = ['train_loss', 'train_accuracy', 'step', 'epoch']
    val_metrics = ['val_loss', 'val_accuracy', 'epoch']

    df_train_metrics = df.loc[:, train_metrics]
    df_train_metrics = df_train_metrics.dropna()

    df_train_metrics_by_epoch = df_train_metrics.groupby(['epoch'])[['train_loss', 'train_accuracy']].mean()

    df_val_metrics = df.loc[:, val_metrics]
    df_val_metrics = df_val_metrics.dropna(how='all', subset=['val_loss', 'val_accuracy'])
    df_val_metrics = df_val_metrics.set_index('epoch')

    df_metrics = pd.concat([df_train_metrics_by_epoch, df_val_metrics], axis=1, join='inner')

    df_metrics['epoch'] = df_metrics.index

    df_metrics.index = [0] * len(df_metrics)

#     df_all = pd.concat([df_metrics, df_hparams], axis=1, join='outer')

    return df_metrics


def get_one_run_dataframe(yaml_file_path: str):
    with open(yaml_file_path) as file:
        hparams = yaml.load(file, Loader=yaml.Loader)
    
    filename = os.path.basename(yaml_file_path)
    run_version, model, _ = filename.split('.')
    hparams = {'cfg': hparams['cfg'].__class__.__module__ + '.' + hparams['cfg'].__class__.__name__, 
               'run': run_version, 
               **hparams['cfg'].__dict__}

    df_hparams = pd.DataFrame.from_dict([hparams])    
    
    df_metrics = from_csv_to_dataframe(yaml_file_path.replace('yaml', 'csv'))

    df_all = pd.concat([df_metrics, df_hparams], axis=1, join='outer')
    
    return df_all
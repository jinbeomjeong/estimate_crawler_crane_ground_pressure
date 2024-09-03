import os
import pytorch_lightning as pl
from utils.pressue_dataset_loader import PressureDataModule
from utils.pressure_est_model import LoadEstModel
from multiprocessing import freeze_support
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


if __name__ == '__main__':
    freeze_support()

    csv_logger = CSVLogger(save_dir="logs", name="create_est_pressure", version=1)

    seq_len = 20
    pred_distance = 50

    train_data_file_path_list = ['safe-boom-40-swing-180-load-40-.csv', 'safe-boom-50-swing-180-load-50-.csv',
                                 'safe-boom-60-swing-180-load-60-.csv', 'safe-boom-70-swing-180-load-100-.csv',
                                 'unsafe-swing-0-load-70-.csv', 'unsafe-swing-0-load-90-.csv',
                                 'unsafe-swing-45-load-50-.csv', 'unsafe-swing-45-load-70-.csv',
                                 'unsafe-swing-90-load-70-.csv', 'unsafe-swing-90-load-90-.csv',
                                 'unsafe-swing-135-load-50-.csv', 'unsafe-swing-135-load-70-.csv']
    val_data_file_path_list = ['safe-boom-80-swing-180-load-120-.csv', 'unsafe-swing-180-load-70-.csv',
                               'unsafe-swing-180-load-90-.csv']


    for i, file_path in enumerate(train_data_file_path_list):
        train_data_file_path_list[i] = os.path.join(file_path)

    for i, file_path in enumerate(val_data_file_path_list):
        val_data_file_path_list[i] = os.path.join(file_path)

    data = PressureDataModule(train_data_path_list=train_data_file_path_list, val_data_path_list=val_data_file_path_list,
                              seq_len=seq_len, pred_distance=pred_distance, batch_size=50, n_of_worker=4)

    model = LoadEstModel(hidden_size=1024, num_layers=1, learning_rate=0.001)

    trainer = pl.Trainer(accelerator='gpu', devices='auto', max_epochs=-1, enable_progress_bar=True, logger=csv_logger,
                         callbacks=EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min'))
    trainer.fit(model=model, datamodule=data)
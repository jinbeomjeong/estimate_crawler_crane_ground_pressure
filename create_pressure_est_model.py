import os
import pytorch_lightning as pl
from utils.pressue_dataset_loader import PressureDataModule
from utils.pressure_est_model import PressureModel
from multiprocessing import freeze_support
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


if __name__ == '__main__':
    freeze_support()

    csv_logger = CSVLogger("logs", name="create_est_pressure", version=2)

    seq_len = 100
    pred_distance = 30

    data_root_path = '.' + os.sep + 'data'
    data_file_name_list = os.listdir(data_root_path)

    data = PressureDataModule(data_path_list=data_file_name_list, seq_len=seq_len, pred_distance=pred_distance,
                              batch_size=500, n_of_worker=8)

    model = PressureModel(hidden_size=1024, num_layers=1, learning_rate=1.0)

    trainer = pl.Trainer(accelerator='gpu', devices='auto', max_epochs=-1, enable_progress_bar=True, logger=csv_logger,
                         callbacks=EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min'))
    trainer.fit(model=model, datamodule=data)
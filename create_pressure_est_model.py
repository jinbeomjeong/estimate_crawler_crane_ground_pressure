import os
import pytorch_lightning as pl
from utils.pressue_dataset_loader import PressureDataModule
from utils.pressure_est_model import PressureModel
from multiprocessing import freeze_support
from pytorch_lightning.loggers import CSVLogger


if __name__ == '__main__':
    freeze_support()

    csv_logger = CSVLogger("logs", name="create_est_pressure", version=1)

    seq_len = 100
    pred_distance = 300

    data_root_path = '.' + os.sep + 'data'
    data_file_name_list = os.listdir(data_root_path)

    data = PressureDataModule(data_path_list=data_file_name_list, seq_len=seq_len, pred_distance=pred_distance,
                              batch_size=200, n_of_worker=8)

    print(data)
    model = PressureModel(hidden_size=1024, num_layers=1, learning_rate=0.001)

    trainer = pl.Trainer(accelerator='cpu', devices='auto', max_epochs=5, enable_progress_bar=True, logger=csv_logger)
    trainer.fit(model=model, datamodule=data)
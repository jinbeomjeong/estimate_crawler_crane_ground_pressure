import asyncio, logging, os
import pandas as pd
import numpy as np

from asyncio import Queue
from pymodbus.server import StartAsyncSerialServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusDeviceContext, ModbusServerContext
from pymodbus import ModbusDeviceIdentification


logging.basicConfig()
_logger = logging.getLogger()
_logger.setLevel(logging.INFO)

store = ModbusDeviceContext(hr=ModbusSequentialDataBlock(address=0, values=[0]*100))
context = ModbusServerContext(devices={1: store}, single=False)
identity = ModbusDeviceIdentification(info_name={"VendorName": "Pymodbus"})


async def data_producer(queue: Queue):
    _logger.info("starting data producer loop")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data = pd.read_csv(os.path.join(root_dir, "test_swing_0_boom_60.csv"))
    time_data = raw_data['Time(sec)'].to_numpy()[::2]
    time_data = time_data.astype(np.uint16)
    max_len = time_data.shape[0]

    front_load = np.abs(raw_data['FL-load'].to_numpy()[::2])*100
    front_load = front_load.astype(np.uint16)

    rear_load = np.abs(raw_data['RR-load'].to_numpy()[::2])*100
    rear_load = rear_load.astype(np.uint16)

    left_load = np.abs(raw_data['FR-load'].to_numpy()[::2])*100
    left_load = left_load.astype(np.uint16)

    right_load = np.abs(raw_data['RR-load'].to_numpy()[::2])*100
    right_load = right_load.astype(np.uint16)

    i = 0

    while True:
        pos = i *4
        var_1 = (1000*np.sin(np.deg2rad(i)))+1000
        var_1 = var_1.astype(np.uint16)

        value_to_send = [var_1.item(), pos, np.abs(10000-pos).item(), front_load[i].item(), rear_load[i].item(), left_load[i].item(),
                         right_load[i].item(), time_data[i].item()]
        await asyncio.sleep(0.1)
        await queue.put(value_to_send)

        i = (i + 1) % max_len


async def modbus_updater(queue: Queue, context: ModbusServerContext):
    _logger.info("starting modbus loop")
    slave_context = context[1]
    register_address = 0

    while True:
        new_value = await queue.get()
        slave_context.setValues(3, register_address, new_value)
        queue.task_done()


async def run_modbus_server(context, identity):
    _logger.info("### initializing modbus server ###")
    await StartAsyncSerialServer(context=context,
                                 identity=identity,
                                 port="/dev/com2",
                                 framer="rtu",
                                 baudrate=115200)


async def main():
    data_queue = Queue()

    server = run_modbus_server(context, identity)
    producer = data_producer(data_queue)
    consumer = modbus_updater(data_queue, context)

    await asyncio.gather(server, producer, consumer)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n Terminate Modbus Server...")

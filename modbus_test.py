import struct
from pymodbus.client import ModbusSerialClient


client = ModbusSerialClient(port='/usb0')
client.connect()


addr = 2  # Modbus 주소 40001
response = client.read_input_registers(address=addr, count=2)
#response = client.read_holding_registers(address=addr, count=2, unit=1)
buffer = response.registers

raw = struct.pack('>HH', buffer[0], buffer[1])
value = struct.unpack('>f', raw)[0]
print(value)

value = 9.87654321
raw = struct.pack('>f', value)
registers = struct.unpack('>HH', raw)
print(registers)

success = client.write_registers(address=4, values=registers)

# 연결 종료
client.close()

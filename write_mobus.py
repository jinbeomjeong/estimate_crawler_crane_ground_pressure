from pymodbus.client import ModbusTcpClient
import struct

# Modbus 서버에 연결
client = ModbusTcpClient('localhost')  # 서버의 IP 주소
address = 2  # Modbus 주소 40001
value = 123.456  # 쓰려는 float32 값

raw = struct.pack('>f', value)
registers = struct.unpack('>HH', raw)
response = client.write_registers(address, re, unit=1)


# 홀딩 레지스터에 값을 쓰는 함수
def write_holding_registers(client, address, values):
    try:
        # 주소에서 values 값을 씀
        response = client.write_registers(address, values, unit=1)
        if not response.isError():
            return True
        else:
            print(f"Error writing holding registers: {response}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False


# Float32 값을 쓰는 함수
def write_float32(client, address, value):
    try:
        # 32비트 float 값을 두 개의 16비트 레지스터 값으로 변환
        raw = struct.pack('>f', value)
        registers = struct.unpack('>HH', raw)

        # 변환된 값을 홀딩 레지스터에 씀
        success = write_holding_registers(client, address, registers)
        return success
    except Exception as e:
        print(f"Exception: {e}")
        return False


# 홀딩 레지스터 주소 40001 (Modbus 주소로는 0)에 float32 값 쓰기
address = 2  # Modbus 주소 40001
value = 123.456  # 쓰려는 float32 값

success = write_float32(client, address, 0.0)

if success:
    print(f"Successfully wrote float32 value: {value}")
else:
    print("Failed to write float32 value")

# 연결 종료
client.close()
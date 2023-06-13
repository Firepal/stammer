import numpy as np

def get_string_from_double(num) -> str:
    parts = num.hex().split('0x1.')[-1].split('p')
    hex_mantissa = '1' + parts[0]
    unshifted = bin(int(hex_mantissa,16))[2:]
    place = int(parts[1])
    if place > 0:
        unshifted = unshifted + place*'0'
    elif place < 0:
        unshifted = (-1*place)*'0' + unshifted
    return unshifted.ljust(64, '0')

def as_array(num):
    string = get_string_from_double(num)
    bit_list = [int(x) for x in string]
    arr = np.array(bit_list)
    hot_bits = np.nonzero(arr)[0]
    return bit_list, hot_bits

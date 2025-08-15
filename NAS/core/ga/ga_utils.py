from graycode import gray_code

FLOAT_TO_INT_CONVERT_VAL = 1000000

def get_gray_code(val):
    was_float = False

    if type(val) == float:
        val *= FLOAT_TO_INT_CONVERT_VAL
        val = int(val)
        was_float = True

    converted_val = bin(gray_code.tc_to_gray_code(val))
    clear_gray_code = converted_val[2:]

    return clear_gray_code, was_float

def convert_to_int(val, was_float):  # Rename
    val = '0b' + val
    val = int(val, 2)
    decoded_val = gray_code.gray_code_to_tc(val)
    if was_float is True:
        decoded_val = float(decoded_val)
        decoded_val /= FLOAT_TO_INT_CONVERT_VAL

    return decoded_val
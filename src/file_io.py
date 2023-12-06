def read_from_bin_file(file_path):
    with open(file_path, 'rb') as file:
        # Read the padding information
        padding_info_byte = file.read(1)
        padding_length = int.from_bytes(padding_info_byte, byteorder='big')

        # Read the rest of the file
        encoded_data = file.read()

        # Convert to a binary string, removing the padding at the end
        bit_string = ''.join([f'{byte:08b}' for byte in encoded_data])[:-padding_length]

        return bit_string


def read_from_txt_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_to_bin_file(file_path, encoded_text): #writes encoded text to a binary file
    # Pad encoded_text to be multiple of 8 
    # as the byte is least addresable memory unit for binary files
    extra_padding = 8 - len(encoded_text) % 8
    for i in range(extra_padding):
        encoded_text += "0"

    # Keep track of padding length
    padded_info = "{0:08b}".format(extra_padding)
    encoded_text = padded_info + encoded_text

    # Convert to bytes and write to file
    with open(file_path, 'wb') as file:
        for i in range(0, len(encoded_text), 8):
            byte = encoded_text[i:i+8] #extracting byte chunks
            file.write(bytes([int(byte, 2)]))

def write_to_txt_file(file_path, decoded_text):
    with open(file_path, 'w') as file:
        file.write(decoded_text)
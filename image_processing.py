'''
This module provides a variety of preprocessing functions for bitmaps designed to run on the ESP-32 Micropython environment.   

Not all of them are necessarily useful for the end product of a neural network model that has learned from images, but experimenting 
with them can go some way to understanding what can be done with the limited resources at your disposal.

The functions are clearly named for what they do. 

Currently the images are generally converted from 96 x 9x pixel image, which is the smallest resolution available from the OV2640 camera 
, captured in bitmap mode and 8 bit grayscale.  To resize to 32x32, for the learning process is then simply done by taking every 3rd pixel.
More "complex" conversions are availalbe (like averaging) but they tend to take a lot longer than we would like.  Converting an image for 
look up in our CNN model at frame rate would be idea. 

You can experiment with this module, by running the socket_server.py Python script after setting the parameters (like wifi password and mode)


'''

from array import array
import math

# This reduces a 96x96 bitmap to 32x32 and applies a threshold conversion to it, converting
# Any pixels greater than or equal to the threshold will become white, while any images less than
# the threshold become black. This makes the image simpler for training a neural network
# altough obviously some accuracy is lost.
# setting the inversion parameter to True will reverse the black and white
# specifying a threshold < 0 will prevent the threshold from being applied and image
# will be full grayscale, not monochrome
# note that the bmp_data_mv parameter includes the bitmap headers


def resize_96x96_to_32x32_and_threshold(bmp_data_mv, threshold, inversion = False):

    bmp_data = bytearray(bmp_data_mv)
    OLD_WIDTH = 96
    OLD_HEIGHT = 96
    NEW_WIDTH = 32
    NEW_HEIGHT = 32

    OLD_BMP_HEADER_SIZE = 14
    OLD_DIB_HEADER_SIZE = 40
    OLD_PALETTE_SIZE = 256 * 4

    NEW_BMP_HEADER_SIZE = 14
    NEW_DIB_HEADER_SIZE = 40
    NEW_PALETTE_SIZE = 256 * 4
    NEW_ROW_PADDING = (NEW_WIDTH % 4)  # Padding for each row

    new_file_size = (
        NEW_BMP_HEADER_SIZE +
        NEW_DIB_HEADER_SIZE +
        NEW_PALETTE_SIZE +
        (NEW_WIDTH + NEW_ROW_PADDING) * NEW_HEIGHT
    )

    new_bmp_data = bytearray(new_file_size)

    # Copy the palette
    palette_offset = OLD_BMP_HEADER_SIZE + OLD_DIB_HEADER_SIZE
    new_palette_offset = NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE
    new_bmp_data[new_palette_offset:new_palette_offset + OLD_PALETTE_SIZE] = \
        bmp_data[palette_offset:palette_offset + OLD_PALETTE_SIZE]

    # Fill headers for the new BMP
    new_bmp_data[0:2] = b'BM'  # Signature
    new_bmp_data[2:6] = new_file_size.to_bytes(4, 'little')  # File size
    new_bmp_data[10:14] = (NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE + NEW_PALETTE_SIZE).to_bytes(4, 'little')  # Data offset
    new_bmp_data[14:18] = NEW_DIB_HEADER_SIZE.to_bytes(4, 'little')  # DIB header size
    new_bmp_data[18:22] = NEW_WIDTH.to_bytes(4, 'little')  # Width
    new_bmp_data[22:26] = NEW_HEIGHT.to_bytes(4, 'little')  # Height
    new_bmp_data[26:28] = b'\x01\x00'  # Planes
    new_bmp_data[28:30] = b'\x08\x00'  # Bits per pixel
    new_bmp_data[34:38] = ((NEW_WIDTH + NEW_ROW_PADDING) * NEW_HEIGHT).to_bytes(4, 'little')  # Image size

    old_pixel_data_offset = OLD_BMP_HEADER_SIZE + OLD_DIB_HEADER_SIZE + OLD_PALETTE_SIZE
    new_pixel_data_offset = NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE + NEW_PALETTE_SIZE

    for new_y in range(NEW_HEIGHT):
        for new_x in range(NEW_WIDTH):
            # Map new pixel coordinates to old coordinates
            old_x = (new_x * OLD_WIDTH) // NEW_WIDTH
            old_y = (new_y * OLD_HEIGHT) // NEW_HEIGHT
            old_y = OLD_HEIGHT - 1 - old_y  # Reverse the row order
            old_pixel_offset = old_pixel_data_offset + old_y * (OLD_WIDTH + (OLD_WIDTH % 4)) + old_x

            # Interpret the byte as unsigned (MicroPython handles this automatically)
            pixel_value = bmp_data[old_pixel_offset] & 0xFF #
            # Apply threshold if > 0, otherwise just keep all color values.
            if threshold >= 0:
                if inversion:
                    pixel_value = 0 if pixel_value >= threshold else 255
                else:    
                    pixel_value = 255 if pixel_value >= threshold else 0

            # Write the pixel to the new BMP
            new_bmp_data[new_pixel_data_offset] = pixel_value
            new_pixel_data_offset += 1

        # Handle row padding for the new BMP
        new_pixel_data_offset += NEW_ROW_PADDING

    return new_bmp_data

# Resizes to 32 x 32 and then quantizes the image to have "depth" colors (grayscale).
# This can simplify the data in an image, but may also cause loss of detail and edges by merging
# otherwise contrasting lines. 

def resize_96x96_to_32x32_quantized(bmp_data_mv, depth):
    if depth < 2: #ignore fewer than 2 colors or a negative depth
        depth = 256             
    bmp_data = bytearray(bmp_data_mv)
    OLD_WIDTH = 96
    OLD_HEIGHT = 96
    NEW_WIDTH = 32
    NEW_HEIGHT = 32

    OLD_BMP_HEADER_SIZE = 14
    OLD_DIB_HEADER_SIZE = 40
    OLD_PALETTE_SIZE = 256 * 4

    NEW_BMP_HEADER_SIZE = 14
    NEW_DIB_HEADER_SIZE = 40
    NEW_PALETTE_SIZE = 256 * 4
    NEW_ROW_PADDING = (NEW_WIDTH % 4)  # Padding for each row

    new_file_size = (
        NEW_BMP_HEADER_SIZE +
        NEW_DIB_HEADER_SIZE +
        NEW_PALETTE_SIZE +
        (NEW_WIDTH + NEW_ROW_PADDING) * NEW_HEIGHT
    )

    new_bmp_data = bytearray(new_file_size)
    range_size = 256 // depth
    




    # Copy the palette
    palette_offset = OLD_BMP_HEADER_SIZE + OLD_DIB_HEADER_SIZE
    new_palette_offset = NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE
    new_bmp_data[new_palette_offset:new_palette_offset + OLD_PALETTE_SIZE] = \
        bmp_data[palette_offset:palette_offset + OLD_PALETTE_SIZE]

    # Fill headers for the new BMP
    new_bmp_data[0:2] = b'BM'  # Signature
    new_bmp_data[2:6] = new_file_size.to_bytes(4, 'little')  # File size
    new_bmp_data[10:14] = (NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE + NEW_PALETTE_SIZE).to_bytes(4, 'little')  # Data offset
    new_bmp_data[14:18] = NEW_DIB_HEADER_SIZE.to_bytes(4, 'little')  # DIB header size
    new_bmp_data[18:22] = NEW_WIDTH.to_bytes(4, 'little')  # Width
    new_bmp_data[22:26] = NEW_HEIGHT.to_bytes(4, 'little')  # Height
    new_bmp_data[26:28] = b'\x01\x00'  # Planes
    new_bmp_data[28:30] = b'\x08\x00'  # Bits per pixel
    new_bmp_data[34:38] = ((NEW_WIDTH + NEW_ROW_PADDING) * NEW_HEIGHT).to_bytes(4, 'little')  # Image size

    old_pixel_data_offset = OLD_BMP_HEADER_SIZE + OLD_DIB_HEADER_SIZE + OLD_PALETTE_SIZE
    new_pixel_data_offset = NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE + NEW_PALETTE_SIZE

    for new_y in range(NEW_HEIGHT):
        for new_x in range(NEW_WIDTH):
            # Map new pixel coordinates to old coordinates
            old_x = (new_x * OLD_WIDTH) // NEW_WIDTH
            old_y = (new_y * OLD_HEIGHT) // NEW_HEIGHT
            old_y = OLD_HEIGHT - 1 - old_y  # Reverse the row order
            old_pixel_offset = old_pixel_data_offset + old_y * (OLD_WIDTH + (OLD_WIDTH % 4)) + old_x

            pixel_value = bmp_data[old_pixel_offset] & 0xFF            
            # Compute the quantized level
            quantized_level = pixel_value // range_size
            # Map the quantized level to the center of its range
            pixel_value = (quantized_level * range_size) + range_size // 2   
            # Write the pixel to the new BMP
            new_bmp_data[new_pixel_data_offset] = pixel_value
            new_pixel_data_offset += 1

        # Handle row padding for the new BMP
        new_pixel_data_offset += NEW_ROW_PADDING

    return new_bmp_data

# Resizes to 32 x 32 and then makes the image black and white by applying thresholding.
# Any pixels with value below the threshold will be 0 (black) and any others will be 255 (white).
# If inversion is True, then the colors are inverted so that values below the threshold become 255 (white) etc.

def resize_96x96_to_32x32_averaged_and_threshold(bmp_data_mv, threshold, inversion=False):
    bmp_data = bytearray(bmp_data_mv)
    OLD_WIDTH = 96
    OLD_HEIGHT = 96
    NEW_WIDTH = 32
    NEW_HEIGHT = 32

    OLD_BMP_HEADER_SIZE = 14
    OLD_DIB_HEADER_SIZE = 40
    OLD_PALETTE_SIZE = 256 * 4

    NEW_BMP_HEADER_SIZE = 14
    NEW_DIB_HEADER_SIZE = 40
    NEW_PALETTE_SIZE = 256 * 4
    NEW_ROW_PADDING = (NEW_WIDTH % 4)  # Padding for each row

    new_file_size = (
        NEW_BMP_HEADER_SIZE +
        NEW_DIB_HEADER_SIZE +
        NEW_PALETTE_SIZE +
        (NEW_WIDTH + NEW_ROW_PADDING) * NEW_HEIGHT
    )

    new_bmp_data = bytearray(new_file_size)

    # Copy the palette
    palette_offset = OLD_BMP_HEADER_SIZE + OLD_DIB_HEADER_SIZE
    new_palette_offset = NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE
    new_bmp_data[new_palette_offset:new_palette_offset + OLD_PALETTE_SIZE] = \
        bmp_data[palette_offset:palette_offset + OLD_PALETTE_SIZE]

    # Fill headers for the new BMP
    new_bmp_data[0:2] = b'BM'  # Signature
    new_bmp_data[2:6] = new_file_size.to_bytes(4, 'little')  # File size
    new_bmp_data[10:14] = (NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE + NEW_PALETTE_SIZE).to_bytes(4, 'little')  # Data offset
    new_bmp_data[14:18] = NEW_DIB_HEADER_SIZE.to_bytes(4, 'little')  # DIB header size
    new_bmp_data[18:22] = NEW_WIDTH.to_bytes(4, 'little')  # Width
    new_bmp_data[22:26] = NEW_HEIGHT.to_bytes(4, 'little')  # Height
    new_bmp_data[26:28] = b'\x01\x00'  # Planes
    new_bmp_data[28:30] = b'\x08\x00'  # Bits per pixel
    new_bmp_data[34:38] = ((NEW_WIDTH + NEW_ROW_PADDING) * NEW_HEIGHT).to_bytes(4, 'little')  # Image size

    old_pixel_data_offset = OLD_BMP_HEADER_SIZE + OLD_DIB_HEADER_SIZE + OLD_PALETTE_SIZE
    new_pixel_data_offset = NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE + NEW_PALETTE_SIZE

    # Resize using pixel averaging
    scale_factor = OLD_WIDTH // NEW_WIDTH
    row_stride = OLD_WIDTH + (OLD_WIDTH % 4)

    for new_y in range(NEW_HEIGHT):
        for new_x in range(NEW_WIDTH):
            # Compute the block of old pixels to average
            old_x_start = new_x * scale_factor
            old_y_start = (NEW_HEIGHT - 1 - new_y) * scale_factor  # Reverse the row order
            old_x_end = old_x_start + scale_factor
            old_y_end = old_y_start + scale_factor
            pixel_sum = 0
            pixel_count = 0

            for old_y in range(old_y_start, old_y_end):
                for old_x in range(old_x_start, old_x_end):
                    old_pixel_offset = old_pixel_data_offset + old_y * row_stride + old_x
                    pixel_sum += bmp_data[old_pixel_offset]
                    pixel_count += 1

            # Average the pixel values
            avg_pixel_value = pixel_sum // pixel_count

            # Apply threshold
            if inversion:
                pixel_value = 0 if avg_pixel_value >= threshold else 255
            else:
                pixel_value = 255 if avg_pixel_value >= threshold else 0

            # Write the pixel to the new BMP
            new_bmp_data[new_pixel_data_offset] = pixel_value
            new_pixel_data_offset += 1

        # Handle row padding for the new BMP
        new_pixel_data_offset += NEW_ROW_PADDING

    return new_bmp_data




#This does Sobel Edge detection (you can look up the algorithm) on an image, copying it into the output image array.
#Note that the input image may include a bitmap header, in which case the offset parameter should point to
#The start of the actual pixel data. The output data is an array just big enough to hold the pixel data.
#Size fixed to 32 x 32.

def sobel_edge_detection(input_image, output_image, offset):
    # Define flattened Sobel kernels
    Gx = array('b', [-1,  0,  1, -2,  0,  2, -1,  0,  1])
    Gy = array('b', [-1, -2, -1,  0,  0,  0,  1,  2,  1])
    
    # Image dimensions
    width = 32
    height = 32

    # Iterate through the image (excluding border pixels)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Apply Sobel kernels using flattened arrays
            sum_x = 0
            sum_y = 0
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    # Adjust the pixel index by adding the offset
                    pixel_index = offset + (y + ky) * width + (x + kx)
                    pixel = input_image[pixel_index]
                    kernel_index = (ky + 1) * 3 + (kx + 1)  # Index into the 1D kernel
                    sum_x += pixel * Gx[kernel_index]
                    sum_y += pixel * Gy[kernel_index]
                    #print (f"ky is {ky}, kx is {kx} pixel_index is {pixel_index}, pixel = {pixel}, kernel index is {kernel_index} Gx is {pixel * Gx[kernel_index]} sum_x is {sum_x}, sum_y is {sum_y}")
            # Compute gradient magnitude
            magnitude = math.sqrt(sum_x**2 + sum_y**2)
            if magnitude < 255:
                magnitude = 0
            # Clamp the magnitude to [0, 255] and store it in the output image
            output_image[y * width + x] = 0 if magnitude < 255 else 255
                # Pad the edges with zeroes
    for y in range(height):
        for x in range(width):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                output_image[y * width + x] = 0  # Zero out the edges
                
                

# This converts a 96 x 96 px bitmap image to a 32 x 32 px bitmap image.
# It uses very simple "every 3rd pixel" conversion, which is why it is reduced to 32 x 32
# and not 28 x 28 which might be more common for some AI learning libraries. 
# note that the bmp_data_mv parameter includes the bitmap headers

def resize_96x96_to_32x32(bmp_data_mv):

    bmp_data = bytearray(bmp_data_mv)
    print(f"bmp_data type is {type(bmp_data)}")
    OLD_WIDTH = 96
    OLD_HEIGHT = 96
    NEW_WIDTH = 32
    NEW_HEIGHT = 32

    OLD_BMP_HEADER_SIZE = 14
    OLD_DIB_HEADER_SIZE = 40
    OLD_PALETTE_SIZE = 256 * 4

    NEW_BMP_HEADER_SIZE = 14
    NEW_DIB_HEADER_SIZE = 40
    NEW_PALETTE_SIZE = 256 * 4
    NEW_ROW_PADDING = (NEW_WIDTH % 4)  # Padding for each row

    new_file_size = (
        NEW_BMP_HEADER_SIZE +
        NEW_DIB_HEADER_SIZE +
        NEW_PALETTE_SIZE +
        (NEW_WIDTH + NEW_ROW_PADDING) * NEW_HEIGHT
    )

    new_bmp_data = bytearray(new_file_size)

    # Copy the palette
    palette_offset = OLD_BMP_HEADER_SIZE + OLD_DIB_HEADER_SIZE
    new_palette_offset = NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE
    new_bmp_data[new_palette_offset:new_palette_offset + OLD_PALETTE_SIZE] = \
        bmp_data[palette_offset:palette_offset + OLD_PALETTE_SIZE]

    # Fill headers for the new BMP
    new_bmp_data[0:2] = b'BM'  # Signature
    new_bmp_data[2:6] = new_file_size.to_bytes(4, 'little')  # File size
    new_bmp_data[10:14] = (NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE + NEW_PALETTE_SIZE).to_bytes(4, 'little')  # Data offset
    new_bmp_data[14:18] = NEW_DIB_HEADER_SIZE.to_bytes(4, 'little')  # DIB header size
    new_bmp_data[18:22] = NEW_WIDTH.to_bytes(4, 'little')  # Width
    new_bmp_data[22:26] = NEW_HEIGHT.to_bytes(4, 'little')  # Height
    new_bmp_data[26:28] = b'\x01\x00'  # Planes
    new_bmp_data[28:30] = b'\x08\x00'  # Bits per pixel
    new_bmp_data[34:38] = ((NEW_WIDTH + NEW_ROW_PADDING) * NEW_HEIGHT).to_bytes(4, 'little')  # Image size

    old_pixel_data_offset = OLD_BMP_HEADER_SIZE + OLD_DIB_HEADER_SIZE + OLD_PALETTE_SIZE
    new_pixel_data_offset = NEW_BMP_HEADER_SIZE + NEW_DIB_HEADER_SIZE + NEW_PALETTE_SIZE

    for new_y in range(NEW_HEIGHT):
        for new_x in range(NEW_WIDTH):
            # Map new pixel coordinates to old coordinates
            old_x = (new_x * OLD_WIDTH) // NEW_WIDTH
            old_y = (new_y * OLD_HEIGHT) // NEW_HEIGHT
            old_y = OLD_HEIGHT - 1 - old_y  # Reverse the row order
            old_pixel_offset = old_pixel_data_offset + old_y * (OLD_WIDTH + (OLD_WIDTH % 4)) + old_x

            # Interpret the byte as unsigned (MicroPython handles this automatically)
            pixel_value = bmp_data[old_pixel_offset] & 0xFF #
            # Write the pixel to the new BMP
            new_bmp_data[new_pixel_data_offset] = pixel_value
            new_pixel_data_offset += 1

        # Handle row padding for the new BMP
        new_pixel_data_offset += NEW_ROW_PADDING

    return new_bmp_data


#
#    Strips the header from a 32x32, 8-bit BMP file and returns only the pixel data.
#    Args:
#    bmp_byte_array (bytes): The byte array of the BMP file, including the header.
#    Returns:
#    bytes: A byte array containing only the pixel data.

def strip_bmp_header(bmp_byte_array):
    BMP_HEADER_SIZE = 54  # Standard BMP header size
    PALETTE_SIZE = 256 * 4  # 256 colors, each with 4 bytes (RGBA)

    # Validate the BMP file size
    if len(bmp_byte_array) <= BMP_HEADER_SIZE + PALETTE_SIZE:
        raise ValueError("Invalid BMP file: insufficient data for header, palette, and pixel data.")

    # Strip the header and color palette
    pixel_data_start = BMP_HEADER_SIZE + PALETTE_SIZE
    pixel_data = bmp_byte_array[pixel_data_start:]

    # Validate the pixel data size
    if len(pixel_data) != 32 * 32:
        raise ValueError("Invalid BMP file: pixel data size is not 32x32.")

    # Quantize the pixel data: 0 → 0, values > 0 → 1
    #quantized_pixel_data = bytearray((1 if pixel > 0 else 0) for pixel in pixel_data)

    # Return the quantized pixel data
    return pixel_data

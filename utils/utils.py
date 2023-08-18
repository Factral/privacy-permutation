import random

import matplotlib.pyplot as plt
import numpy as np
import torch


class Permutar():
    def __init__(self, image_size, block_size):
        self.block_size = block_size
        self.image_size = image_size
        self.blocks_row = (self.image_size // self.block_size)
        self.key = np.argsort(np.random.rand((self.blocks_row ** 2)))

    def get_start_block(self, block):
        n_row = block // self.blocks_row
        n_col = block % self.blocks_row
        return np.array([n_row * self.block_size, n_col * self.block_size])

    def desordenar(self, pic):
        image = pic.clone()
        for place, replace in enumerate(self.key):
            s_blk = self.get_start_block(place)
            r_blk = self.get_start_block(replace)

            image[:, :, s_blk[0]:s_blk[0] + self.block_size, s_blk[1]:s_blk[1] + self.block_size] = pic[:, :,
                                                                                                    r_blk[0]:r_blk[
                                                                                                                 0] + self.block_size,
                                                                                                    r_blk[1]:r_blk[
                                                                                                                 1] + self.block_size]
        return image

    def ordenar(self, pic):
        image = pic.clone()
        for place, replace in enumerate(self.key):
            s_blk = self.get_start_block(place)
            r_blk = self.get_start_block(replace)

            image[:, :, r_blk[0]:r_blk[0] + self.block_size, r_blk[1]:r_blk[1] + self.block_size] = pic[:, :, s_blk[0]:s_blk[0] + self.block_size,s_blk[1]:s_blk[1] + self.block_size]
        return image

    def ordenar_feature_map(self, feature_map):
        image = feature_map.clone()
        for place, replace in enumerate(self.key):
            s_blk = self.get_start_block(place) // self.block_size
            r_blk = self.get_start_block(replace) // self.block_size

            image[:, :, r_blk[0], r_blk[1]] = feature_map[:, :, s_blk[0], s_blk[1]]
        return image

    @staticmethod
    def plot(self, image):
        image = image - torch.min(image)  # undo normalization
        image = image / torch.max(image)
        plt.imshow(image.permute(1, 2, 0))
        plt.show()

    def __call__(self, image):
        return self.desordenar(image)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


def get_index_conv(index, size):
    # this will return in a list, the index of all pixels that are in the convolution
    # index is the pixel target, the center of the convolution, like [5,5]
    # the convolution is size x size

    index_range = range(-(size // 2), size // 2 + 1)
    return [[index[0] + i, index[1] + j] for i in index_range for j in index_range]


def calculate_index_conv(dims, stride, padding, kernel_size):
    # Calcula el tamaño de la salida de la convolución
    output_dims = (dims + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    # Crea una matriz para almacenar los índices de los píxeles centrales
    center_indices = np.zeros((output_dims, output_dims, 2), dtype=int)

    # Calcula los índices de los píxeles centrales
    center_indices[:, :, 0] = np.arange(-padding + kernel_size // 2, dims + padding - kernel_size // 2, stride)[:,
                              np.newaxis]
    
    center_indices[:, :, 1] = np.arange(-padding + kernel_size // 2, dims + padding - kernel_size // 2, stride)[
                              np.newaxis, :]

    return center_indices

def calculate_index_pool(dims, stride, padding, kernel_size):
    # Calcula el tamaño de la salida de el pool
    output_dims = (dims + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    # Crea una matriz para almacenar los índices de los píxeles centrales
    center_indices = np.zeros((output_dims, output_dims, 2), dtype=int)

    # Calcula los índices de los píxeles centrales
    center_indices[:, :, 0] = np.arange(-padding + kernel_size // 2 - 1, dims + padding - kernel_size // 2, stride)[:,
                              np.newaxis]
    
    center_indices[:, :, 1] = np.arange(-padding + kernel_size // 2 - 1, dims + padding - kernel_size // 2, stride)[
                              np.newaxis, :]

    return center_indices


def get_index_pool(index, size):
    # this will return in a list, the index of all pixels that are in the pool
    # index is the pixel target, the left top corner of the pool, like [0,0]
    # the pool is size x size

    index_range = range(size)
    return [[index[0] + i, index[1] + j] for i in index_range for j in index_range]

def get_pixels_from_coordinates(input, coordinates):
    # coordinates is a list like this: [[0, 0], [0, 1], [1, 0], [1, 1]]
    # input is a tensor of shape (batch, channels, height, width)
    # this function returns a tensor of shape (batch, channels, len(coordinates))

    batch_size, channels, height, width = input.shape
    pixels = torch.zeros((batch_size, channels, len(coordinates)), dtype=input.dtype, device=input.device)

    for i, coordinate in enumerate(coordinates):
        pixels[:, :, i] = input[:, :, coordinate[0], coordinate[1]]

    return pixels


def calculate_offset(dims, stride, padding, kernel_size, permutation, batch=64):
    output_dims = (dims + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    # generate key for the permutation of conv
    perm = Permutar(output_dims, 1)
    index_positions = calculate_index_conv(dims, stride, padding, kernel_size)

    # reorder index position
    index_positions = np.transpose(index_positions, (2, 0, 1))
    print(index_positions.shape)

    # desdordenar conv
    index_positions_ = perm.desordenar(torch.from_numpy(index_positions).unsqueeze(0)).squeeze(0).numpy()

    pixel_positions = np.array([permutation.get_start_block(xi) for xi in permutation.key])

    offset = np.zeros((2 * kernel_size ** 2, output_dims, output_dims), dtype='float32')

    for i in range(output_dims):
        for j in range(output_dims):
            positions = get_index_conv(index_positions_[:, i, j], kernel_size)
            positions_ = get_index_conv(index_positions[:, i, j], kernel_size)

            for a, pos in enumerate(positions):
                index = np.where((pixel_positions == pos).all(axis=1))[0]
                if index.size > 0:
                    fake_pos = permutation.get_start_block(index)
                    distance = np.squeeze(fake_pos) - positions_[a]

                    offset[a * 2, i, j] = distance[0]
                    offset[a * 2 + 1, i, j] = distance[1]

    offset = np.tile(offset, [batch, 1, 1, 1])
    return torch.from_numpy(offset).float(), perm



def deform_maxPool2d(input,kernel_size,stride,padding):
    dims = input.shape[-1]

    output_dims = (dims + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    index_positions = calculate_index_pool(dims,stride,padding,kernel_size)
    index_positions = np.transpose(index_positions, (2, 0, 1))
        
    batch_size, channels, height, width = input.shape

    result = torch.zeros(batch_size,channels,output_dims,output_dims)

    for i in range(output_dims):
        for j in range(output_dims):
            positions = get_index_pool(index_positions[:, i, j], kernel_size)
            pixels = get_pixels_from_coordinates(input, positions)
            result[:,:,i,j] = torch.amax(pixels,2)
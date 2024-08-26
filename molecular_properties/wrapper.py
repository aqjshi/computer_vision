import numpy as np
import re
import pandas as pd




def heat_component(tensor, tensor_x, tensor_y, tensor_z, tensor_weight, resolution_dim):
    # Determine the base indices
    base_x, base_y, base_z = int(tensor_x), int(tensor_y), int(tensor_z)
    # Get the decimal part for bleeding
    bleed_x = tensor_x % 1
    bleed_y = tensor_y % 1
    bleed_z = tensor_z % 1

    complement_bleed_x = 1 - bleed_x
    complement_bleed_y = 1 - bleed_y
    complement_bleed_z = 1 - bleed_z

    # Calculate the weights for each of the neighboring cells
    weights = np.zeros((2, 2, 2))
    weights[0, 0, 0] = complement_bleed_x * complement_bleed_y * complement_bleed_z * tensor_weight
    weights[0, 0, 1] = complement_bleed_x * complement_bleed_y * bleed_z * tensor_weight
    weights[0, 1, 0] = complement_bleed_x * bleed_y * complement_bleed_z * tensor_weight
    weights[0, 1, 1] = complement_bleed_x * bleed_y * bleed_z * tensor_weight
    weights[1, 0, 0] = bleed_x * complement_bleed_y * complement_bleed_z * tensor_weight
    weights[1, 0, 1] = bleed_x * complement_bleed_y * bleed_z * tensor_weight
    weights[1, 1, 0] = bleed_x * bleed_y * complement_bleed_z * tensor_weight
    weights[1, 1, 1] = bleed_x * bleed_y * bleed_z * tensor_weight

    # Add weights to the tensor
    for dx in range(2):
        for dy in range(2):
            for dz in range(2):
                if 0 <= base_x + dx < tensor.shape[0] and 0 <= base_y + dy < tensor.shape[1] and 0 <= base_z + dz < tensor.shape[2]:
                    tensor[base_x + dx, base_y + dy, base_z + dz] += weights[dx, dy, dz]

    return tensor



def read_data(filename):
    data = np.load(filename, allow_pickle=True)
    df = pd.DataFrame(data.tolist() if data.dtype == 'O' and isinstance(data[0], dict) else data)
    return df

def npy_preprocessor_v4(filename):
    df = read_data(filename)
    return df['index'].values, df['inchi'].values, df['xyz'].values, df['chiral_centers'].values, df['rotation'].values

def npy_preprocessor_v4_limit(filename, limit=10):
    df = read_data(filename)
    return df['index'].values[:limit], df['inchi'].values[:limit], df['xyz'].values[:limit], df['chiral_centers'].values[:limit], df['rotation'].values[:limit]
def npy_preprocessor(filename):
    df = read_data(filename)
    return df['index'].values, df['xyz'].values, df['rotation'].values




import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive image creation
import matplotlib.pyplot as plt

static_iamge_size = 27
def decimal_to_bw_image(values, image_size=27):
    """
    Converts an array of decimal values into a black and white image.

    Parameters:
        values (list or np.array): A list or array of decimal values.
        image_size (int): The size of the image (default is 27x27).

    Returns:
        None: Displays the black and white image.
    """
    # Ensure values is a numpy array
    values = np.array(values)

    num_pixels = image_size * image_size

    if len(values) != num_pixels:
        raise ValueError(f"Expected {num_pixels} values but got {len(values)}")

    # Normalize the values between 0 and 1
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))

    # Reshape the array into an image_size x image_size matrix
    image_matrix = normalized_values.reshape(image_size, image_size)

    # Display the image
    plt.imshow(image_matrix, cmap='gray', interpolation='nearest')
    plt.axis('off')  # Turn off axis labels
    plt.show()




def decimal_to_rgb_image(values, image_size=27, output_dir='images', filename='output_image.png'):
    """
    Converts an array of decimal values into an RGB image and saves it as a PNG file.

    Parameters:
        values (list or np.array): A list or array of decimal values.
        image_size (int): The size of the image (default is 27x27).
        output_dir (str): Directory to save the image.
        filename (str): Name of the output image file.

    Returns:
        None: Saves the image as a PNG file.
    """
    # Ensure values is a numpy array
    values = np.array(values)

    num_pixels = image_size * image_size
    expected_num_values = num_pixels * 3  # For RGB image

    if len(values) != expected_num_values:
        raise ValueError(f"Expected {expected_num_values} values but got {len(values)}")

    # Normalize the values between 0 and 1, then scale to [0, 255]
    normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))
    scaled_values = (normalized_values * 255).astype(np.uint8)  # Scale and convert to 8-bit integer

    # Reshape the array into num_pixels tuples of 3 values (R, G, B)
    rgb_tuples = scaled_values.reshape(num_pixels, 3)

    # Reshape the RGB tuples into an image_size x image_size x 3 matrix
    image_matrix = rgb_tuples.reshape(image_size, image_size, 3)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create the full path for the output image
    image_path = os.path.join(output_dir, filename)

    # Save the image as a PNG
    plt.imshow(image_matrix, interpolation='nearest')
    plt.axis('off')  # Turn off axis labels
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()



def read_values_from_file(file_path, expected_values=None):
    """
    Reads decimal values from a text file.

    Parameters:
        file_path (str): Path to the text file.
        expected_values (int): Expected number of values (optional).

    Returns:
        np.array: Array of decimal values.
    """
    with open(file_path, 'r') as file:
        values = list(map(float, file.read().strip().split()))
    
    if expected_values is not None and len(values) != expected_values:
        raise ValueError(f"Expected {expected_values} values but got {len(values)}")
    
    return np.array(values)




def read_voxel_data(file_path):
    """
    Read voxel data from a file and return as a list of numpy arrays.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if not lines:
            return []
        
        return [np.array(list(map(float, line.strip().split()))) for line in lines]


def preprocess_limited_voxels(directory, num_voxels):
    """
    Preprocess a limited number of voxel files in the specified directory.
    """
    files = sorted(os.listdir(directory))[:num_voxels]
    voxel_list = []

    for file in files:
        file_path = os.path.join(directory, file)
        voxel_data = read_voxel_data(file_path)
        
        if voxel_data:
            voxel_list.append(voxel_data[0])  # Use the first voxel in the file
        else:
            print(f"Warning: Skipped file {file_path} due to empty or malformed data.")
    
    return voxel_list


def preprocess_all_voxels(directory):
    """
    Preprocess all voxel files in the specified directory and return combined data.
    """
    combined_voxel_data = []
    files = sorted(os.listdir(directory))

    for file in files:
        file_path = os.path.join(directory, file)
        voxel_data = read_voxel_data(file_path)
        
        if voxel_data:
            combined_voxel_data.extend(voxel_data[0])
        else:
            print(f"Warning: Skipped file {file_path} due to empty or malformed data.")
    
    return np.array(combined_voxel_data)


def preprocess_data(data_dir, num_voxels, image_size=27):
    """
    Preprocess a specified number of voxel data files from the directory.

    Parameters:
        data_dir (str): Directory containing voxel data files.
        num_voxels (int): Number of voxel data files to process.
        image_size (int): Size of the image (image_size x image_size).

    Returns:
        np.array: Array of preprocessed voxel data.
    """
    print("Starting voxel data preprocessing...")
    voxel_data = preprocess_limited_voxels(data_dir, num_voxels)
    
    num_pixels = image_size * image_size
    expected_values = num_pixels * 3  # For RGB images

    # Pad or truncate the data to ensure each has expected_values
    adjusted_voxel_data = []
    for voxel in voxel_data:
        voxel = np.resize(voxel, expected_values)
        adjusted_voxel_data.append(voxel)
    
    adjusted_voxel_data = np.array(adjusted_voxel_data)
    
    print(f"Preprocessing complete. Number of voxel images loaded: {len(adjusted_voxel_data)}")
    return adjusted_voxel_data

def voxel_to_image(voxel_data, image_size=27):
    """
    Convert voxel data to an image_size x image_size RGB image.

    Parameters:
        voxel_data (np.array): 1D array of voxel data values.
        image_size (int): Size of the output image (image_size x image_size).

    Returns:
        np.array: image_size x image_size x 3 RGB image.
    """
    num_channels = 3
    total_voxels = len(voxel_data) // num_channels

    # Calculate grid size N such that N^3 = total_voxels
    N = int(round(total_voxels ** (1/3)))

    if N ** 3 != total_voxels:
        raise ValueError(f"Cannot reshape data into a cube of size {N}^3")

    # Split voxel data into R, G, B channels
    r_channel = voxel_data[0::3].reshape((N, N, N))
    g_channel = voxel_data[1::3].reshape((N, N, N))
    b_channel = voxel_data[2::3].reshape((N, N, N))

    # Normalize the channels to [0, 255]
    r_channel = (r_channel * 255).astype(np.uint8)
    g_channel = (g_channel * 255).astype(np.uint8)
    b_channel = (b_channel * 255).astype(np.uint8)

    # Flatten the 3D grid into a 1D array
    r_flat = r_channel.flatten()
    g_flat = g_channel.flatten()
    b_flat = b_channel.flatten()

    # Calculate the total number of pixels in the image
    total_pixels = image_size * image_size

    # Ensure the flattened arrays have the correct number of pixels
    # Truncate or pad as necessary
    r_flat = np.resize(r_flat, total_pixels)
    g_flat = np.resize(g_flat, total_pixels)
    b_flat = np.resize(b_flat, total_pixels)

    # Reshape into image_size x image_size
    r_image = r_flat.reshape(image_size, image_size)
    g_image = g_flat.reshape(image_size, image_size)
    b_image = b_flat.reshape(image_size, image_size)

    # Stack the channels to form an image_size x image_size x 3 RGB image
    return np.stack((r_image, g_image, b_image), axis=-1)



def extract_patches(voxel_data, image_size=27, patch_size=9):
    """
    Extract patches from a voxel image.

    Parameters:
        voxel_data (np.array): 1D array of voxel data values.
        image_size (int): Size of the image (image_size x image_size).
        patch_size (int): Size of the patches.

    Returns:
        list of np.array: List of patches extracted from the image.
    """
    reshaped_data = voxel_data.reshape((image_size, image_size, 3))
    patches = []

    for i in range(0, reshaped_data.shape[0], patch_size):
        for j in range(0, reshaped_data.shape[1], patch_size):
            patch = reshaped_data[i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    
    return patches


def visualize_voxel_and_patches(voxel_data, image_title, image_size=27, patch_size=9):
    """
    Visualize the voxel image and its patches.

    Parameters:
        voxel_data (np.array): 1D array of voxel data values.
        image_title (str): Title for the image.
        image_size (int): Size of the image (image_size x image_size).
        patch_size (int): Size of the patches.

    Returns:
        None
    """
    # Visualize the original voxel image
    image = voxel_to_image(voxel_data, image_size=image_size)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(image_title)
    plt.axis('off')
    plt.show()

    # Extract and visualize patches
    patches = extract_patches(voxel_data, image_size=image_size, patch_size=patch_size)
    num_patches = min(10, len(patches))

    fig_rows = (num_patches + 4) // 5  # Up to 5 patches per row
    fig, axes = plt.subplots(fig_rows, 5, figsize=(15, 3 * fig_rows))
    
    for idx in range(num_patches):
        if fig_rows > 1:
            ax = axes[idx // 5, idx % 5]
        else:
            ax = axes[idx % 5]
        ax.imshow(patches[idx])
        ax.set_title(f'Patch {idx+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()




def convert_file_to_image(file_path, output_dir='images', image_size=27):
    """
    Converts a text file containing values into an RGB image.

    Parameters:
        file_path (str): Path to the text file.
        output_dir (str): Directory to save the image.
        image_size (int): Size of the image.

    Returns:
        None: Saves the image as a JPG file in the specified directory.
    """
    # Extract the filename without extension to use as the output image name
    filename = os.path.splitext(os.path.basename(file_path))[0] + '.jpg'

    # Calculate expected number of values
    num_pixels = image_size * image_size
    expected_num_values = num_pixels * 3  # For RGB image

    # Read the values from the file
    values = read_values_from_file(file_path, expected_values=expected_num_values)

    # Convert the values to an RGB image and save it
    decimal_to_rgb_image(values, image_size=image_size, output_dir=output_dir, filename=filename)

                        

def process_and_visualize(data_dir, num_voxels, image_index=0):
    """
    Process and visualize a specific voxel image.
    """
    files = sorted(os.listdir(data_dir))
    
    if image_index >= len(files):
        print(f"Image index {image_index} out of range. Using the first image instead.")
        image_index = 0
    
    selected_file = files[image_index]
    image_title = os.path.splitext(selected_file)[0]

    # Preprocess and visualize the voxel data
    voxel_data = preprocess_data(data_dir, num_voxels)
    selected_voxel = voxel_data[image_index]
    visualize_voxel_and_patches(selected_voxel, image_title)


# # Example usage
if __name__ == "__main__":
    file_path = "test_rs/000044$1$S$0$0.txt"  # Replace with the path to your file
    output_dir = "final_images"
    image_size = 27  # Example of a different image size

    convert_file_to_image(file_path, output_dir, image_size=image_size)

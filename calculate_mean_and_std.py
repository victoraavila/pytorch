import torch

def calculate_mean_and_std(dataset_data: torch.Tensor):
    num_images = len(dataset_data)
    num_pixels_per_image = len(dataset_data[0].flatten())

    sum_of_pixel_values = 0
    for img_tensor in dataset_data:
        flattened_data = img_tensor.flatten() / 255.0
        sum_of_pixel_values += torch.sum(flattened_data)

    mean = sum_of_pixel_values / (num_images * num_pixels_per_image)

    sum_of_squared_diffs = 0
    for img_tensor in dataset_data:
        flattened_data = img_tensor.flatten() / 255.0
        # Using the mean of the entire dataset to make only one division
        # Squaring it in order to remove negatives
        sum_of_squared_diffs += torch.sum((flattened_data - mean) ** 2)

    variance = sum_of_squared_diffs / (num_images * num_pixels_per_image)
    std = torch.sqrt(variance)
    
    return mean, std
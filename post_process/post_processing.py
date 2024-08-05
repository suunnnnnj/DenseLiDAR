import cv2
import numpy as np
import os


def load_depth_image(path):
    # Load depth image with any depth
    depth_image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)

    if depth_image is None:
        raise FileNotFoundError(f"Image at path {path} not found.")

    depth_image = depth_image.astype(np.float32) / 256.0

    return depth_image


def calculate_rmse(predicted_depth, pseudo_depth, outlier_mask):
    # Calculate RMSE excluding outliers
    valid_points = ~outlier_mask

    if np.sum(valid_points) == 0:
        return np.nan

    rmse = np.sqrt(np.nanmean((predicted_depth[valid_points] - pseudo_depth[valid_points]) ** 2))

    return rmse


def remove_outliers(predicted_depth, pseudo_depth):
    # Calculate depth difference
    depth_difference = np.abs(predicted_depth - pseudo_depth)

    # Threshold settings [(max_distance,threshold)]
    thresholds = [(10, 0.07), (30, 0.08), (np.inf, 0.1)]

    # Create outlier mask
    outlier_mask = np.zeros_like(depth_difference, dtype=bool)

    # Apply thresholds
    for max_depth, threshold in thresholds:
        range_mask = (pseudo_depth <= max_depth)
        outlier_condition = (depth_difference > threshold)
        outlier_mask |= outlier_condition & range_mask

        num_outliers = np.sum(outlier_condition & range_mask)
        print(f"Threshold: {threshold} | Max Depth: {max_depth} | Outliers in range: {num_outliers}")

    # Remove outliers
    post_processing_depth = np.where(~outlier_mask, predicted_depth, np.nan)

    return post_processing_depth, outlier_mask


def print_unique_values(image, image_name):
    unique_values = np.unique(image)
    print(f"Unique values in {image_name}: {unique_values}")


def main(predicted_depth_path, pseudo_depth_path):
    # Load images
    predicted_depth = load_depth_image(predicted_depth_path)
    pseudo_depth = load_depth_image(pseudo_depth_path)

    # Ensure both images have the same shape
    if predicted_depth.shape != pseudo_depth.shape:
        raise ValueError("Predicted depth and pseudo depth images must have the same dimensions.")

    # Print unique pixel values
    print_unique_values(predicted_depth, "predicted_depth")
    print_unique_values(pseudo_depth, "pseudo_depth")

    # Remove outliers
    post_processing_depth, outlier_mask = remove_outliers(predicted_depth, pseudo_depth)

    # Calculate RMSE
    rmse = calculate_rmse(post_processing_depth, pseudo_depth, outlier_mask)

    # Output results
    retained_percentage = np.sum(~outlier_mask) / outlier_mask.size * 100
    print(f"RMSE after outlier removal: {rmse:.2f} mm")
    print(f"Percentage of points retained: {retained_percentage:.2f}%")

    # Save post processing depth
    post_processing_depth_path = "post_processing_depth.png"
    cv2.imwrite(post_processing_depth_path, post_processing_depth)
    print(f"Post processing depth image saved to {post_processing_depth_path}")

    import matplotlib.pyplot as plt
    plt.imshow(post_processing_depth, 'gray')
    plt.show()


dir_path = os.path.dirname(os.path.realpath(__file__))
# Example usage
predicted_depth_path = os.path.join(dir_path, 'dense_depth_output.png')
pseudo_depth_path = os.path.join(dir_path, 'demo_pseudo_depth.png')
main(predicted_depth_path, pseudo_depth_path)

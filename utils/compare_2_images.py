import os
import cv2
import matplotlib.pyplot as plt


ground_truth_folder = r"C:\Users\prasa\Downloads\rrlab_download\simulated\gt_labels"
predicted_folder = r'C:\Users\prasa\Downloads\rrlab_download\simulated\prediction'

# Example file names (replace with actual file names or loop through files)
gt_filename = '1721394108_Camera0_class.png'
pred_filename = '1721394108_Camera0_visible.png'

# Read the ground truth and predicted images
gt_image_path = os.path.join(ground_truth_folder, gt_filename)
pred_image_path = os.path.join(predicted_folder, pred_filename)

gt_image = cv2.imread(gt_image_path)
pred_image = cv2.imread(pred_image_path)

# Convert the images from BGR to RGB (as OpenCV loads images in BGR by default)
gt_image_rgb = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
pred_image_rgb = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)

# Check if the images were loaded successfully
if gt_image_rgb is None:
    print(f"Error: Could not load ground truth image: {gt_image_path}")
elif pred_image_rgb is None:
    print(f"Error: Could not load predicted image: {pred_image_path}")
else:
    # Plot both images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(gt_image_rgb)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')  # Hide axis

    axes[1].imshow(pred_image_rgb)
    axes[1].set_title("Prediction")
    axes[1].axis('off')  # Hide axis

    plt.tight_layout()
    plt.show()

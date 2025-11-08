import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.segmentation import slic, mark_boundaries
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_distances
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.stats import kendalltau, spearmanr
import sys

# Load the pre-trained ResNet18 model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# Define the image preprocessing transformations
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the ImageNet class index mapping
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
idx2synset = [class_idx[str(k)][0] for k in range(len(class_idx))]
id2label = {v[0]: v[1] for v in class_idx.values()}

imagenet_path = "./imagenet_samples"

# List of image file paths
image_paths = os.listdir(imagenet_path)

method = sys.argv[1] if len(sys.argv) > 1 else "unknown"


def predict(imgs):
    batch_tensors = []
    for input_image in imgs:
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image.astype(np.uint8))
        input_tensor = preprocess(input_image)
        batch_tensors.append(input_tensor)
    batch = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.cpu().numpy()


def lime_explanation(image_paths):

    predictions = []

    # Configuration
    NUM_TOP_CLASSES_LIME = 3
    NUM_FEATURES_TO_SHOW_LIME = 10

    predictions = []

    for img_path in image_paths:
        # Load and preprocess
        full_path = os.path.join(imagenet_path, img_path)
        input_image = Image.open(full_path).convert("RGB")
        img_np = np.array(input_image)

        # Get model prediction probabilities
        probs = predict([input_image])[0]

        # Get top K classes
        top_k_indices = np.argsort(probs)[-NUM_TOP_CLASSES_LIME:][::-1]
        top_k_probs = probs[top_k_indices]
        top_k_labels = [idx2label[idx] for idx in top_k_indices]

        print(f"\nProcessing: {img_path}")
        print(f"Top {NUM_TOP_CLASSES_LIME} predictions:")
        for i, (idx, prob, label) in enumerate(
            zip(top_k_indices, top_k_probs, top_k_labels)
        ):
            print(f"  {i+1}. {label}: {prob:.3f}")

        predictions.append((img_path, top_k_indices, top_k_labels, top_k_probs))

        # Segment image
        segments = slic(img_np, n_segments=100, compactness=15, sigma=1)
        num_segments = len(np.unique(segments))

        # Generate perturbations
        num_samples = 1000
        perturbations = np.random.randint(0, 2, size=(num_samples, num_segments))
        imgs_perturbed = []

        for mask in perturbations:
            temp = img_np.copy()
            off_segments = np.where(mask == 0)[0]
            avg_color = img_np.mean(axis=(0, 1))
            for seg_val in off_segments:
                temp[segments == seg_val] = avg_color
            imgs_perturbed.append(temp.astype(np.uint8))

        # Predict on perturbed images
        preds_perturbed = predict(imgs_perturbed)

        # Compute similarity weights (same for all classes)
        original_instance = np.ones((1, num_segments))
        distances = np.sqrt(
            np.sum(perturbations**2, axis=1)[:, np.newaxis]
            + np.sum(original_instance**2, axis=1)[np.newaxis, :]
            - 2 * np.dot(perturbations, original_instance.T)
        ).ravel()
        kernel_width = 0.25 * np.sqrt(num_segments)
        weights = np.exp(-(distances**2) / (kernel_width**2))

        # Train one explainer per top class
        explainers = {}
        for class_idx in top_k_indices:
            y = preds_perturbed[:, class_idx]

            clf = Ridge(alpha=1.0)

            clf.fit(perturbations, y, sample_weight=weights)
            explainers[class_idx] = clf.coef_

        # Create visualization
        img_float = img_np / 255.0
        num_cols = NUM_TOP_CLASSES_LIME + 2  # original + segmentation + one per class
        fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

        # Original image
        axes[0].imshow(img_float)
        axes[0].set_title("Original Image", fontsize=10)
        axes[0].axis("off")

        # Segmentation
        axes[1].imshow(mark_boundaries(img_float, segments))
        axes[1].set_title(f"Superpixels\n({num_segments} segments)", fontsize=10)
        axes[1].axis("off")

        # Explanation for each top class
        for plot_idx, (class_idx, prob, label) in enumerate(
            zip(top_k_indices, top_k_probs, top_k_labels)
        ):
            coefficients = explainers[class_idx]

            # Get top features by absolute importance
            top_features_idx = np.argsort(np.abs(coefficients))[
                -NUM_FEATURES_TO_SHOW_LIME:
            ]

            # Separate into positive and negative
            positive_features = [
                idx for idx in top_features_idx if coefficients[idx] > 0
            ]
            negative_features = [
                idx for idx in top_features_idx if coefficients[idx] < 0
            ]

            # Build masks
            mask_positive = np.isin(segments, positive_features)
            mask_negative = np.isin(segments, negative_features)

            # Create color overlays
            overlay = np.zeros_like(img_float)
            overlay[..., 1][mask_positive] = 1.0  # green for positive
            overlay[..., 0][mask_negative] = 1.0  # red for negative

            # Blend
            alpha = 0.5
            highlighted = (1 - alpha) * img_float + alpha * overlay

            # Plot
            ax = axes[plot_idx + 2]
            ax.imshow(highlighted)
            ax.set_title(
                f"Class {plot_idx + 1}: {label}\n(prob: {prob:.3f})", fontsize=10
            )
            ax.axis("off")

        plt.suptitle(
            "LIME Explanation (Green=Positive, Red=Negative)",
            fontsize=12,
            y=1.02,
        )
        plt.tight_layout()

        # Save
        if not os.path.exists("lime_outputs"):
            os.makedirs("lime_outputs")
        output_path = os.path.join(
            "lime_outputs", f"{os.path.splitext(img_path)[0]}_lime_explanation.JPEG"
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()

    return predictions


if method == "lime":
    lime_outputs = lime_explanation(image_paths)
    exit(0)


def smoothgrad(img_tensor, target_class):
    grad_sum = torch.zeros_like(img_tensor).to(device)
    for _ in range(NUM_SAMPLES_SG):
        noisy_img = img_tensor + (
            torch.randn_like(img_tensor).to(device) * NOISE_LEVEL_SG
        )
        noisy_img = noisy_img.clone().detach().requires_grad_(True)
        outputs = model(noisy_img)
        target_score = outputs[0, target_class]
        model.zero_grad()
        target_score.backward()
        grad_sum += noisy_img.grad.data

    smooth_grad = grad_sum / NUM_SAMPLES_SG
    return smooth_grad.squeeze(0).cpu().numpy()


def visualize_smoothgrad(img_np, smooth_grads, title):
    # Resize original image to match attribution size
    img_resized = np.array(Image.fromarray(img_np).resize((224, 224)))
    img_float = img_resized / 255.0
    _, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(img_float)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for idx, smooth_grad in enumerate(smooth_grads):
        # Take absolute value and average across color channels
        attribution = np.mean(np.abs(smooth_grad[0]), axis=0)

        # Normalize to [0, 1] using robust percentile method
        vmin = np.percentile(attribution, 1)
        vmax = np.percentile(attribution, 99)

        if vmax - vmin > 0:
            attribution_normalized = (attribution - vmin) / (vmax - vmin)
            attribution_normalized = np.clip(attribution_normalized, 0, 1)
        else:
            attribution_normalized = np.zeros_like(attribution)

        axes[idx + 1].imshow(attribution_normalized, cmap="gray", vmin=0, vmax=1)
        axes[idx + 1].set_title(
            f"label: {smooth_grads[idx][1]}\nprob: {smooth_grads[idx][2]:.3f}"
        )
        axes[idx + 1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()


def smoothgrad_expln(img_path):
    full_path = os.path.join(imagenet_path, img_path)
    input_image = Image.open(full_path).convert("RGB")

    # Get predictions
    probs = predict([input_image])[0]
    top_k_indices = np.argsort(probs)[-NUM_TOP_CLASSES_SG:][::-1]
    top_k_probs = probs[top_k_indices]
    top_k_labels = [idx2label[idx] for idx in top_k_indices]

    print(f"\nProcessing: {img_path}, {NUM_TOP_CLASSES_SG} predictions:")
    for i, (idx, prob, label) in enumerate(
        zip(top_k_indices, top_k_probs, top_k_labels)
    ):
        print(f"  {i+1}. {label}: {prob:.3f}")

    smooth_grads = []
    for class_idx, prob, label in zip(top_k_indices, top_k_probs, top_k_labels):
        img_tensor = preprocess(input_image).unsqueeze(0).to(device)
        smooth_grad = smoothgrad(img_tensor, class_idx)
        smooth_grads.append((smooth_grad, label, prob))

    title = f"SmoothGrad explanations for top {NUM_TOP_CLASSES_SG} classes"
    img_np = np.array(input_image)
    visualize_smoothgrad(img_np, smooth_grads, title)

    if not os.path.exists("smoothgrad_outputs"):
        os.makedirs("smoothgrad_outputs")
    output_path = os.path.join(
        "smoothgrad_outputs",
        f"{os.path.splitext(img_path)[0]}.JPEG",
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_path}")
    plt.close()
    return smooth_grads


# Configuration
NUM_TOP_CLASSES_SG = 3
NUM_SAMPLES_SG = 1000
NOISE_LEVEL_SG = 0.40

if method == "smoothgrad":
    for img_path in image_paths:
        smoothgrad_output = smoothgrad_expln(
            img_path,
        )
    exit(0)

print("Invalid method. Use 'lime' or 'smoothgrad'.")

import os
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Import classes and functions from utils.py
from utils import GeneratorUNet, unnormalize, get_transforms

def generate_comparative_image(image_path, model_path, output_dir):
    """
    Takes a concatenated dataset image, processes it, runs inference using a trained model,
    and saves a comparative plot (Input | Generated | Real) to the specified directory.
    
    Args:
        image_path (str): Relative path to the original concatenated image.
        model_path (str): Relative path to the trained .pth generator model.
        output_dir (str): Directory where the output plot will be saved.
    """
    # Set device dynamically (Supports NVIDIA CUDA, Apple Silicon MPS, or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and split the image (Left: Real, Right: Label Map)
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return
        
    w, h = image.size
    w_half = w // 2
    
    # As defined in MapsDataset (utils): Left is Target, Right is Input
    real_pil = image.crop((0, 0, w_half, h))
    map_pil = image.crop((w_half, 0, w, h))
    
    # Apply standard Pix2Pix transforms from utils.py (resize to 256x256, normalize to [-1, 1])
    transform = get_transforms()
    
    input_tensor = transform(map_pil).unsqueeze(0).to(device)
    real_tensor = transform(real_pil).to(device)
    
    # Load the trained GeneratorUNet
    generator = GeneratorUNet().to(device)
    
    # Load weights
    generator.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    generator.eval()
    
    # Run inference
    with torch.no_grad():
        generated_tensor = generator(input_tensor)
        
    # Unnormalize and move to CPU for matplotlib plotting
    map_viz = unnormalize(input_tensor[0].cpu()).permute(1, 2, 0)
    gen_viz = unnormalize(generated_tensor[0].cpu()).permute(1, 2, 0)
    real_viz = unnormalize(real_tensor.cpu()).permute(1, 2, 0)
    
    # Create Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(map_viz)
    axes[0].set_title("Input Map")
    axes[0].axis("off")
    
    axes[1].imshow(gen_viz)
    axes[1].set_title("Generated Image (Output)")
    axes[1].axis("off")
    
    axes[2].imshow(real_viz)
    axes[2].set_title("Real Image (Target)")
    axes[2].axis("off")
    
    plt.tight_layout()
    
    # Save Plot
    base_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, f"comparison_{base_name}")
    plt.savefig(save_path, bbox_inches='tight')
    
    # Close figure to free memory
    plt.close(fig) 
    print(f"Success! Comparative image saved at: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pix2Pix Inference Demo")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image (concatenated dataset format).")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained generator model (.pth).")
    parser.add_argument("--output", type=str, default="./results", help="Directory to save the result.")
    
    args = parser.parse_args()
    
    generate_comparative_image(args.image, args.model, args.output)
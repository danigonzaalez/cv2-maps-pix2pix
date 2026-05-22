import os
import random
import time
import json
import matplotlib.pyplot as plt
from IPython.display import clear_output
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as TF


# ---------------------------------------------------------
# ---------------------------------------------------------
# DATASET: LOADING AND PREPROCESSING
# ---------------------------------------------------------
# ---------------------------------------------------------

class MapsDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment = False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            augment (Boolean, optional): Optional apply data augmentation on the training set.
        """
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.augment = augment
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        
        # The images in the dataset are concatenated horizontally.
        # Left side: Real Image (Target)
        # Right side: Label Map (Input)
        w, h = image.size
        w_half = w // 2
        
        target_image = image.crop((0, 0, w_half, h))   # left half of the image
        input_map = image.crop((w_half, 0, w, h))   # right half of the image
        
        if self.transform:
            # We fix the random seed just before applying the transforms
            # to ensure that if any random geometric transformations (like flips) 
            # are applied, they are exactly the same for both the input and the target.
            
            # Generate a random seed for this specific sample
            seed = random.randint(0, 2**32)
            
            # Apply transform to input_map
            torch.manual_seed(seed)
            random.seed(seed)
            input_map = self.transform(input_map)
            
            # Apply the exact same transform to target_image
            torch.manual_seed(seed)
            random.seed(seed)
            target_image = self.transform(target_image)

        if self.augment:
            # Include data augmentation

            # Horizontal Flip 
            if random.random() > 0.5:
                input_map = TF.hflip(input_map)
                target_image = TF.hflip(target_image)
            
            # Vertical Flip 
            if random.random() > 0.5:
                input_map = TF.vflip(input_map)
                target_image = TF.vflip(target_image)

            # Rotation 
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                input_map = TF.rotate(input_map, angle)
                target_image = TF.rotate(target_image, angle)
            
        return input_map, target_image

def get_transforms():
    """
    Returns the standard transforms for Pix2Pix.
    Images are resized to 256x256 and normalized to the range [-1, 1].
    Images must be on range [-1, 1] to be compatible with the tanh activation function of the generator.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def unnormalize(tensor):
    """
    Reverts the normalization to display the images correctly.
    """
    return tensor * 0.5 + 0.5


# ---------------------------------------------------------
# ---------------------------------------------------------
# GENERATOR (U-NET) COMPONENTS
# ---------------------------------------------------------
# ---------------------------------------------------------

class UNetDown(nn.Module):
    """
    Encoder block: Downsamples the image by half its width and height.
    """
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        # kernel=4, stride=2, padding=1 exactly halves the spatial dimensions (e.g., 256x256 -> 128x128)
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]

        # If batch normalization is desired
        if normalize:
            # Batch normalization stabilizes training. We skip it on the very first layer.
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(nn.LeakyReLU(0.2))
        
        # If dropout is desired
        if dropout:
            layers.append(nn.Dropout(dropout))
        
        # Define model unpacking layers of the list
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """
    Decoder block: Upsamples the image to double its width and height.
    """
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        # ConvTranspose2d does the exact opposite of Conv2d, doubling spatial dimensions
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        # If dropout is desired
        if dropout:
            # Dropout in the decoder adds randomness to the GAN generation
            layers.append(nn.Dropout(dropout))

        # Define model unpacking layers of the list
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        # 1. Upsample the input from the previous layer
        x = self.model(x)
        
        # 2. Skip Connection: Concatenate the upsampled output with the corresponding feature map 
        # from the encoder. We concatenate along the channel dimension (dim=1).
        # E.g., if x has 512 channels and skip_input has 512 channels, the output has 1024 channels.
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    """
    The full U-Net architecture bridging the Down and Up blocks.
    """
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # ENCODER: Compressing the 256x256 image down to a 1x1 bottleneck (in_channels, out_channels)
        self.down1 = UNetDown(in_channels, 64, normalize=False) # Out: 64x128x128
        self.down2 = UNetDown(64, 128)                          # Out: 128x64x64
        self.down3 = UNetDown(128, 256)                         # Out: 256x32x32
        self.down4 = UNetDown(256, 512, dropout=0.5)            # Out: 512x16x16
        self.down5 = UNetDown(512, 512, dropout=0.5)            # Out: 512x8x8
        self.down6 = UNetDown(512, 512, dropout=0.5)            # Out: 512x4x4
        self.down7 = UNetDown(512, 512, dropout=0.5)            # Out: 512x2x2
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5) # Bottleneck: 512x1x1

        # DECODER: Expanding the bottleneck back to 256x256
        # Notice the in_channels for up2 onwards are double the out_channels of the previous layer 
        # due to skip connections concatenating channels
        self.up1 = UNetUp(512, 512, dropout=0.5)                # Out: 512 (upsampled) + 512 (skip d7) = 1024 x2x2
        self.up2 = UNetUp(1024, 512, dropout=0.5)               # Out: 512 + 512 (skip d6) = 1024 x4x4
        self.up3 = UNetUp(1024, 512, dropout=0.5)               # Out: 512 + 512 (skip d5) = 1024 x8x8
        self.up4 = UNetUp(1024, 512, dropout=0.0)               # Out: 512 + 512 (skip d4) = 1024 x16x16
        self.up5 = UNetUp(1024, 256, dropout=0.0)               # Out: 256 + 256 (skip d3) = 512 x32x32
        self.up6 = UNetUp(512, 128, dropout=0.0)                # Out: 128 + 128 (skip d2) = 256 x64x64
        self.up7 = UNetUp(256, 64, dropout=0.0)                 # Out: 64 + 64 (skip d1) = 128 x128x128

        # Final layer brings it back to 3 RGB channels and applies Tanh to map to [-1, 1]
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),   # Takes 128x128x128 and outputs 3x256x256 (original size)
            nn.Tanh()   # maps 3x256x256 generated image to [-1, 1]
        )

    def forward(self, x):
        # Propagate down the encoder, saving each output for the skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7) # Bottleneck
        
        # Propagate up the decoder, passing the previous output and the skip connection
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)


# ---------------------------------------------------------
# ---------------------------------------------------------
# DISCRIMINATOR (PATCHGAN)
# ---------------------------------------------------------
# ---------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # Helper function to create repetitive blocks of Conv -> BatchNorm -> LeakyReLU
        def discriminator_block(in_filters, out_filters, normalization=True):
            # kernel_size=4, stride=2 and padding=1 halves the image dimensions (e.g., 256 -> 128)
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]

            # If batch normalization is desired
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # First layer: input is 6 channels (3 from condition map + 3 from target image)
            # We don't normalize the first layer in PatchGAN

            # Spatial size: 256x256 -> 128x128
            *discriminator_block(in_channels * 2, 64, normalization=False),
            
            # Second layer: 128x128 -> 64x64
            *discriminator_block(64, 128),
            
            # Third layer: 64x64 -> 32x32
            *discriminator_block(128, 256),   # out: 256x32x32
            
            # Fourth layer:
            # We stop downsampling by half (stride=1). The kernel size 4 and padding 1 will just 
            # slightly reduce the spatial dimension from 32x32 to 31x31.
            nn.Conv2d(256, 512, kernel_size=4, padding=1, stride=1),   # out 512x31x31
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final Output layer: Reduces the 512 channels to 1 single channel.
            # Spatial dimension goes from 31x31 to 30x30.
            # The output is a 1x30x30 matrix, not a single probability value
            nn.Conv2d(512, 1, kernel_size=4, padding=1, stride=1)
        )

    def forward(self, condition, image):
        # Pix2Pix is a Contitional GAN. The discriminator must see both the input map
        # and the image (either real or generated) to evaluate if they match.
        # We concatenate them along the channel dimension (dim=1): 3 + 3 = 6 channels.
        model_input = torch.cat((condition, image), 1)
        return self.model(model_input)
    

# ---------------------------------------------------------
# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
# ---------------------------------------------------------


def train_pix2pix_v2(generator, discriminator, train_loader, val_loader, device, 
                  criterion_GAN, criterion_pixelwise, optimizer_G, optimizer_D,
                  epochs=100, lambda_l1=100, patience=20, output_dir="trained_models", model_name="pix2pix"):

    """
    Trains the Pix2Pix model with integrated validation, learning rate scheduling, 
    and early stopping. Best model weights are saved automatically.

    Args:
        generator (nn.Module): The generator network (typically a U-Net).
        discriminator (nn.Module): The discriminator network (typically a PatchGAN).
        train_loader (DataLoader): PyTorch DataLoader for the training set.
        val_loader (DataLoader): PyTorch DataLoader for the validation set.
        device (torch.device): Computation device (e.g., 'cuda' or 'cpu').
        criterion_GAN (nn.Module): Adversarial loss (e.g., BCEWithLogitsLoss).
        criterion_pixelwise (nn.Module): Pixel-level loss (e.g., L1Loss).
        optimizer_G (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_D (torch.optim.Optimizer): Optimizer for the discriminator.
        epochs (int, optional): Maximum number of training epochs. Defaults to 100.
        lambda_l1 (float, optional): Weight for the pixel-wise L1 loss. Defaults to 100.
        patience (int, optional): Number of epochs to wait for improvement before 
            triggering early stopping. Defaults to 20.
        output_dir (str, optional): Directory where the best models will be saved. 
            Defaults to "trained_models".
        model_name (str, optional): Prefix used for the saved model filenames. 
            Defaults to "pix2pix".

    Returns:
        tuple: (generator, discriminator, history) 
            - generator: The trained generator model.
            - discriminator: The trained discriminator model.
            - history (dict): Dictionary containing training and validation loss logs.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting Training: {model_name} on {device}")

    # History dictionary
    history = {
        "train_G": [], "train_D": [],
        "val_G": [], "val_D": []
    }

    # Early Stopping & LR Decay Setup
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # Scheduler: Reduce lr by factor if loss doesn't improve in half early stop patience
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.7, patience=patience//2)
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.7, patience=patience//2)

    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss_D, epoch_loss_G = 0.0, 0.0
        
        generator.train()
        discriminator.train()
        
        for condition_map, real_image in train_loader:
            condition_map, real_image = condition_map.to(device), real_image.to(device)
            
            # --- PHASE 1: DISCRIMINATOR ---
            optimizer_D.zero_grad()
            pred_real = discriminator(condition_map, real_image)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))
            
            fake_image = generator(condition_map)
            pred_fake = discriminator(condition_map, fake_image.detach())
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # --- PHASE 2: GENERATOR ---
            optimizer_G.zero_grad()
            pred_fake_for_G = discriminator(condition_map, fake_image)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G).to(device))
            loss_G_L1 = criterion_pixelwise(fake_image, real_image)
            
            loss_G = loss_G_GAN + (lambda_l1 * loss_G_L1)
            loss_G.backward()
            optimizer_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()

        # --- VALIDATION PHASE ---
        val_epoch_loss_D, val_epoch_loss_G = 0.0, 0.0
        generator.eval()
        discriminator.eval()
        
        with torch.no_grad():
            for val_map, val_real in val_loader:
                val_map, val_real = val_map.to(device), val_real.to(device)
                v_fake = generator(val_map)
                
                v_loss_D = (criterion_GAN(discriminator(val_map, val_real), torch.ones_like(discriminator(val_map, val_real)).to(device)) + 
                            criterion_GAN(discriminator(val_map, v_fake), torch.zeros_like(discriminator(val_map, v_fake)).to(device))) * 0.5
                v_loss_G = criterion_GAN(discriminator(val_map, v_fake), torch.ones_like(discriminator(val_map, v_fake)).to(device)) + (lambda_l1 * criterion_pixelwise(v_fake, val_real))
                
                val_epoch_loss_D += v_loss_D.item()
                val_epoch_loss_G += v_loss_G.item()

        # Metrics 
        avg_train_G, avg_val_G = epoch_loss_G / len(train_loader), val_epoch_loss_G / len(val_loader)
        avg_train_D, avg_val_D = epoch_loss_D / len(train_loader), val_epoch_loss_D / len(val_loader)
        
        history["train_G"].append(avg_train_G); history["val_G"].append(avg_val_G)
        history["train_D"].append(avg_train_D); history["val_D"].append(avg_val_D)

        # LR Scheduling based on Validation Loss G
        scheduler_G.step(avg_val_G)
        scheduler_D.step(avg_val_D)
        current_lr = optimizer_G.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] - train_G_Loss: {avg_train_G:.4f} | train_D_Loss: {avg_train_D:.4f} val_G_Loss: {avg_val_G:.4f} | val_D_Loss: {avg_val_D:.4f} | LR: {current_lr:.6f} |Time: {time.time()-start_time:.2f}s")

        # --- CHECKPOINT & EARLY STOPPING ---
        if avg_val_G < best_val_loss:
            best_val_loss = avg_val_G
            early_stop_counter = 0
            # Save best model
            torch.save(generator.state_dict(), os.path.join(output_dir, f"best_G_{model_name}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(output_dir, f"best_D_{model_name}.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    return generator, discriminator, history

def train_pix2pix_v3(generator, discriminator, train_loader, val_loader, device, 
                     criterion_GAN, criterion_pixelwise, optimizer_G, optimizer_D,
                     epochs=500, lambda_l1=300, step_size=100, save_step=50, 
                     output_dir="trained_models", model_name="pix2pix"):
    """
    Trains the Pix2Pix model with periodic checkpointing, automated history logging, 
    and StepLR decay. This version is optimized for longer training runs.

    Args:
        generator (nn.Module): The generator network.
        discriminator (nn.Module): The discriminator network.
        train_loader (DataLoader): PyTorch DataLoader for the training set.
        val_loader (DataLoader): PyTorch DataLoader for the validation set.
        device (torch.device): Computation device (e.g., 'cuda' or 'cpu').
        criterion_GAN (nn.Module): Adversarial loss function.
        criterion_pixelwise (nn.Module): Pixel-wise loss function (typically L1).
        optimizer_G (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_D (torch.optim.Optimizer): Optimizer for the discriminator.
        epochs (int, optional): Total number of training epochs. Defaults to 500.
        lambda_l1 (int, optional): Weight for the L1 pixel-wise loss. Defaults to 300.
        step_size (int, optional): Period of learning rate decay in epochs. Defaults to 100.
        save_step (int, optional): Interval of epochs to save model checkpoints 
            and JSON history. Defaults to 50.
        output_dir (str, optional): Directory to save models and logs. Defaults to "trained_models".
        model_name (str, optional): Name used for saved files. Defaults to "pix2pix".

    Returns:
        tuple: (generator, discriminator, history)
            - generator: Trained generator model.
            - discriminator: Trained discriminator model.
            - history (dict): Dictionary containing training/validation loss metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, f"history_{model_name}.json")
    print(f"Starting Training: {model_name} on {device}")

    history = {
        "train_G": [], "train_D": [],
        "val_G": [], "val_D": []
    }

    best_val_loss = float('inf')
    
    # Reduce LR by 0.7 factor each step_size epochs
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=step_size, gamma=0.7)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=step_size, gamma=0.7)

    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss_D, epoch_loss_G = 0.0, 0.0
        
        generator.train()
        discriminator.train()
        
        for condition_map, real_image in train_loader:
            condition_map, real_image = condition_map.to(device), real_image.to(device)
            
            # Discriminator
            optimizer_D.zero_grad()
            pred_real = discriminator(condition_map, real_image)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real).to(device))
            
            fake_image = generator(condition_map)
            pred_fake = discriminator(condition_map, fake_image.detach())
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake).to(device))
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # Generator
            optimizer_G.zero_grad()
            pred_fake_for_G = discriminator(condition_map, fake_image)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G).to(device))
            loss_G_L1 = criterion_pixelwise(fake_image, real_image)
            
            loss_G = loss_G_GAN + (lambda_l1 * loss_G_L1)
            loss_G.backward()
            optimizer_G.step()

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()

        # Validation
        val_epoch_loss_D, val_epoch_loss_G = 0.0, 0.0
        generator.eval()
        discriminator.eval()
        
        with torch.no_grad():
            for val_map, val_real in val_loader:
                val_map, val_real = val_map.to(device), val_real.to(device)
                v_fake = generator(val_map)
                
                v_pred_real = discriminator(val_map, val_real)
                v_pred_fake = discriminator(val_map, v_fake)
                
                v_loss_D = (criterion_GAN(v_pred_real, torch.ones_like(v_pred_real).to(device)) + 
                            criterion_GAN(v_pred_fake, torch.zeros_like(v_pred_fake).to(device))) * 0.5
                v_loss_G = criterion_GAN(v_pred_fake, torch.ones_like(v_pred_fake).to(device)) + (lambda_l1 * criterion_pixelwise(v_fake, val_real))
                
                val_epoch_loss_D += v_loss_D.item()
                val_epoch_loss_G += v_loss_G.item()

        # Calculate average metrics per epoch
        avg_train_G, avg_val_G = epoch_loss_G / len(train_loader), val_epoch_loss_G / len(val_loader)
        avg_train_D, avg_val_D = epoch_loss_D / len(train_loader), val_epoch_loss_D / len(val_loader)
        
        history["train_G"].append(avg_train_G); history["val_G"].append(avg_val_G)
        history["train_D"].append(avg_train_D); history["val_D"].append(avg_val_D)

        # Update schedulers
        scheduler_G.step()
        scheduler_D.step()
        
        current_lr = optimizer_G.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{epochs}] - G_Loss: {avg_val_G:.4f} | D_Loss: {avg_val_D:.4f} | LR: {current_lr:.6f} | Time: {time.time()-start_time:.2f}s")

        # Checkpoints
        # Save best val_G loss model
        if avg_val_G < best_val_loss:
            best_val_loss = avg_val_G
            torch.save(generator.state_dict(), os.path.join(output_dir, f"best_G_{model_name}.pth"))
        
        # Save each n_steps epochs
        if (epoch + 1) % save_step == 0:
            torch.save(generator.state_dict(), os.path.join(output_dir, f"epoch_{epoch+1}_G_{model_name}.pth"))
             # Save history every 50 epochs in case kernel died
            with open(history_path, 'w') as f:
                json.dump(history, f)
            print(f"Checkpoint and History saved at epoch {epoch+1}")

    return generator, discriminator, history

def plot_training_history(history, earlystopping = None):

    """
    Plots the training and validation loss curves for both the Generator and Discriminator.

    This function visualizes the evolution of GAN losses over epochs, allowing for 
    comparison between training and validation performance. It can also highlight 
    the best model checkpoint based on the minimum validation loss.

    Args:
        history (dict): A dictionary containing the following keys:
            - "train_G": List of generator training losses.
            - "train_D": List of discriminator training losses.
            - "val_G": List of generator validation losses.
            - "val_D": List of discriminator validation losses.
        earlystopping (bool, optional): If True, draws a vertical line at the epoch 
            with the lowest Generator validation loss, indicating where the best 
            model was likely saved. Defaults to None.

    Returns:
        None: Displays a Matplotlib plot.
    """
    plt.figure(figsize=(12, 6))
    plt.title("Pix2Pix Training History: Generator vs Discriminator Loss")
    
    # Extraer datos del diccionario
    train_G = history["train_G"]
    train_D = history["train_D"]
    val_G = history["val_G"]
    val_D = history["val_D"]
    epochs = range(1, len(train_G) + 1)

    # Plot Train Losses (líneas sólidas)
    plt.plot(epochs, train_G, label="Train G Loss (Adversarial + L1)", color="blue", linestyle="-", linewidth=1.5)
    plt.plot(epochs, train_D, label="Train D Loss (PatchGAN)", color="orange", linestyle="-", linewidth=1.5)
    
    # Plot Validation Losses (líneas discontinuas)
    plt.plot(epochs, val_G, label="Val G Loss", color="blue", linestyle="--", linewidth=1.5, alpha=0.7)
    plt.plot(epochs, val_D, label="Val D Loss", color="orange", linestyle="--", linewidth=1.5, alpha=0.7)
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    

    if earlystopping:
        # Find the epoch-index
        min_val_idx = val_G.index(min(val_G))
        
        # Mark the bets model saved according to val G loss
        plt.axvline(x=min_val_idx + 1, color='red', linestyle='--', alpha=0.4, label='Best Model')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# ---------------------------------------------------------
# VISUALIZATION AND TEST UTILITY 
# ---------------------------------------------------------
# ---------------------------------------------------------



def model_benchmark(generator, dataset, device, num_samples=3, indexes=None):
    """
    Benchmark fucntion to evaluate the performance of the different models.
    Take indexes of relevant images if given.
    If indexes is None, take random samples with a fixed seed to ensure reproducibility.
    Args:
        generator(nn.Module): Trained generator network.
        dataset(torch.Subset): The subset of images to be tested.
        device(torch.device): The computation device.
        num_samples(int): Number of random samples to be tested.
        indexes(list): List containing the indexes of the images in the dataset to be tested.
    """
    generator.eval()
    
    
    if indexes is None:
        # Fix random seed
        random.seed(42) 
        indexes = [random.randint(0, len(dataset) - 1) for _ in range(num_samples)]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for i, idx in enumerate(indexes):
        # Get the map and image from original dataset
        input_map, real_image = dataset[idx]
        
        # prepare the input 
        input_tensor = input_map.unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated_image = generator(input_tensor)
        
        # Unnormalize  ( [-1, 1] --> [0, 1])
        map_viz = unnormalize(input_map).permute(1, 2, 0).cpu()
        gen_viz = unnormalize(generated_image[0]).permute(1, 2, 0).cpu()
        real_viz = unnormalize(real_image).permute(1, 2, 0).cpu()
        
        # Plot
        axes[i, 0].imshow(map_viz)
        axes[i, 0].set_title(f"Input (Idx {idx})")
        axes[i, 1].imshow(gen_viz)
        axes[i, 1].set_title("Generated (Output)")
        axes[i, 2].imshow(real_viz)
        axes[i, 2].set_title("Real (Target)")
        
        for ax in axes[i]:
            ax.axis("off")
            
    plt.tight_layout()
    plt.show()



def _pypure_ssim(img1, img2, window_size=11):
    """
    Computes the Structural Similarity Index (SSIM) using pure PyTorch.
    Standard implementation with a uniform/Gaussian window approach.
    """
    channel = img1.size(1)

    # Gaussian filter for the local average
    window = torch.ones((channel, 1, window_size, window_size), dtype=img1.dtype, device=img1.device)
    window = window / (window_size * window_size)
    
    # Stability constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Local averages
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # variances and covariances
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def evaluate_test_metrics(generator, test_loader, device):
    """
    Evaluates the generator's performance on the test dataset using L1 loss,
    SSIM, and PSNR. Written in pure PyTorch to avoid library import headaches.
    
    Assumes inputs are normalized in [-1, 1] and unnormalizes them to [0, 1] 
    internally for correct perceptual metric evaluation.
    """
    generator.eval()
    criterion_l1 = nn.L1Loss()
    
    total_test_l1 = 0.0
    total_test_ssim = 0.0
    total_test_psnr = 0.0
    
    print("Running evaluation (L1, SSIM, PSNR) on Test set...")
    
    with torch.no_grad():
        for condition_map, real_image in test_loader:
            condition_map = condition_map.to(device)
            real_image = real_image.to(device)
            
            fake_image = generator(condition_map)
            
            # Compute L1 loss on the original range [-1, 1]
            loss_l1 = criterion_l1(fake_image, real_image)
            total_test_l1 += loss_l1.item()
            
            # dennormalze the iamges to the [0,1] range for SSIM and PSNR
            real_image_unnorm = real_image * 0.5 + 0.5
            fake_image_unnorm = fake_image * 0.5 + 0.5
            fake_image_unnorm = torch.clamp(fake_image_unnorm, 0.0, 1.0)
            
            
             
            mse = F.mse_loss(fake_image_unnorm, real_image_unnorm, reduction='mean')
            if mse.item() == 0:
                psnr = 100.0  # avoid divide into zero
            else:
                psnr = 10 * torch.log10(1.0 / mse)
            total_test_psnr += psnr.item()
            
            
            ssim_val = _pypure_ssim(fake_image_unnorm, real_image_unnorm)
            total_test_ssim += ssim_val.item()
            
    num_batches = len(test_loader)
    avg_test_l1 = total_test_l1 / num_batches
    avg_test_ssim = total_test_ssim / num_batches
    avg_test_psnr = total_test_psnr / num_batches
    
    print("-" * 40)
    print(f"AVERAGE L1 LOSS (TEST): {avg_test_l1:.6f}")
    print(f"AVERAGE SSIM    (TEST): {avg_test_ssim:.4f}")
    print(f"AVERAGE PSNR    (TEST): {avg_test_psnr:.2f} dB")
    print("-" * 40)
    
    return avg_test_l1, avg_test_ssim, avg_test_psnr





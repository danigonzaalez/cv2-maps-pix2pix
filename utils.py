import os
import random
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.optim as optim



# ---------------------------------------------------------
# DATASET: LOADING AND PREPROCESSING
# ---------------------------------------------------------

class MapsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

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
# GENERATOR (U-NET) COMPONENTS
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
# DISCRIMINATOR (PATCHGAN)
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
# TRAINING LOOP
# ---------------------------------------------------------
def train_pix2pix(generator, discriminator, train_loader, val_loader, device, 
                  criterion_GAN, criterion_pixelwise, optimizer_G, optimizer_D,
                  epochs=10, lambda_l1=100):
    """
    Trains the Pix2Pix model, evaluates on the validation set per epoch, 
    and visualizes the results and loss plots at the end of training.
    
    Args:
        generator (nn.Module): The generator network (U-Net) that creates fake images.
        discriminator (nn.Module): The discriminator network (PatchGAN) that classifies images.
        train_loader (DataLoader): PyTorch DataLoader containing the training dataset.
        val_loader (DataLoader): PyTorch DataLoader containing the validation dataset.
        device (torch.device): The computation device to use (e.g., 'cuda' or 'cpu').
        criterion_GAN (nn.Module): Adversarial loss function to evaluate realism (e.g., BCEWithLogitsLoss).
        criterion_pixelwise (nn.Module): Pixel-wise loss function for structural accuracy (e.g., L1Loss).
        optimizer_G (torch.optim.Optimizer): The optimization algorithm for the generator.
        optimizer_D (torch.optim.Optimizer): The optimization algorithm for the discriminator.
        epochs (int, optional): The total number of full passes over the training dataset. Defaults to 10.
        lambda_l1 (float or int, optional): The weight multiplier applied to the pixel-wise loss. Defaults to 100.
        
    Returns:
        tuple: A tuple containing the fully trained (generator, discriminator) models.
    """
    print(f"Starting Training Loop for {epochs} epochs...")

    # Initialize lists to store average epoch losses for both Train and Validation
    train_losses_D, train_losses_G = [], []
    val_losses_D, val_losses_G = [], []

    for epoch in range(epochs):
        start_time = time.time()

        # Variables to accumulate loss for the current epoch (Training)
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        
        # Set models to training mode to enable dropout and batch normalization tracking
        generator.train()
        discriminator.train()
        
        for i, (condition_map, real_image) in enumerate(train_loader):
            # Move data to the active device
            condition_map = condition_map.to(device)
            real_image = real_image.to(device)
            
            # ==================================================
            # PHASE 1: TRAIN DISCRIMINATOR (D)
            # Goal: Maximize the probability of correctly classifying Real and Fake images.
            # ==================================================
            optimizer_D.zero_grad() # Clear previous gradients
            
            # 1. Train with REAL images
            pred_real = discriminator(condition_map, real_image)
            target_real = torch.ones_like(pred_real).to(device) # Real labels are 1s
            loss_real = criterion_GAN(pred_real, target_real)
            
            # 2. Train with FAKE images
            fake_image = generator(condition_map)
            pred_fake = discriminator(condition_map, fake_image.detach())
            target_fake = torch.zeros_like(pred_fake).to(device) # Fake labels are 0s
            loss_fake = criterion_GAN(pred_fake, target_fake)
            
            # 3. Compute total D loss and update weights
            loss_D = (loss_real + loss_fake) * 0.5 
            loss_D.backward()
            optimizer_D.step()
            
            # ==================================================
            # PHASE 2: TRAIN GENERATOR (G)
            # Goal: Minimize the difference between generated and real images (L1), 
            # and fool the discriminator (Adversarial).
            # ==================================================
            optimizer_G.zero_grad() # Clear previous gradients
            
            # 1. Adversarial Loss (Fooling D)
            pred_fake_for_G = discriminator(condition_map, fake_image)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, target_real) 
            
            # 2. Pixel-wise Loss (Structural accuracy)
            loss_G_L1 = criterion_pixelwise(fake_image, real_image)
            
            # 3. Compute total G loss and update weights
            loss_G = loss_G_GAN + (lambda_l1 * loss_G_L1)
            loss_G.backward()
            optimizer_G.step()

            # Add batch loss to epoch accumulators
            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()

        # ==================================================
        # VALIDATION PHASE
        # ==================================================
        # Variables to accumulate loss for the current epoch (Validation)
        val_epoch_loss_D = 0.0
        val_epoch_loss_G = 0.0
        
        # Set models to evaluation mode (turns off dropout, fixes batchnorm)
        generator.eval()
        discriminator.eval()
        
        with torch.no_grad(): # Disable gradient computation
            for val_map, val_real in val_loader:
                val_map = val_map.to(device)
                val_real = val_real.to(device)
                
                # Forward passes for validation
                val_fake_image = generator(val_map)
                
                # Discriminator Validation Loss
                val_pred_real = discriminator(val_map, val_real)
                val_loss_real = criterion_GAN(val_pred_real, torch.ones_like(val_pred_real).to(device))
                
                val_pred_fake = discriminator(val_map, val_fake_image)
                val_loss_fake = criterion_GAN(val_pred_fake, torch.zeros_like(val_pred_fake).to(device))
                
                v_loss_D = (val_loss_real + val_loss_fake) * 0.5
                val_epoch_loss_D += v_loss_D.item()
                
                # Generator Validation Loss
                val_pred_fake_for_G = discriminator(val_map, val_fake_image)
                v_loss_G_GAN = criterion_GAN(val_pred_fake_for_G, torch.ones_like(val_pred_fake_for_G).to(device))
                v_loss_G_L1 = criterion_pixelwise(val_fake_image, val_real)
                
                v_loss_G = v_loss_G_GAN + (lambda_l1 * v_loss_G_L1)
                val_epoch_loss_G += v_loss_G.item()

        # ==================================================
        # END OF EPOCH PROCESSING
        # ==================================================
        epoch_duration = time.time() - start_time
        
        # Calculate average losses for the epoch (Train & Validation)
        avg_train_loss_D = epoch_loss_D / len(train_loader)
        avg_train_loss_G = epoch_loss_G / len(train_loader)
        avg_val_loss_D = val_epoch_loss_D / len(val_loader)
        avg_val_loss_G = val_epoch_loss_G / len(val_loader)
        
        # Append to main lists
        train_losses_D.append(avg_train_loss_D)
        train_losses_G.append(avg_train_loss_G)
        val_losses_D.append(avg_val_loss_D)
        val_losses_G.append(avg_val_loss_G)
        
        # Print epoch summary
        print(f"---- Epoch [{epoch+1}/{epochs}] finished in {epoch_duration:.2f} seconds ----")
        print(f"Train - D Loss: {avg_train_loss_D:.4f} | G Loss: {avg_train_loss_G:.4f}")
        print(f"Val   - D Loss: {avg_val_loss_D:.4f} | G Loss: {avg_val_loss_G:.4f}\n")
            
    # ==================================================
    # END OF TRAINING: PLOT LOSSES & VISUALIZE SAMPLE
    # ==================================================
    
    # 1. Plotting the losses
    plt.figure(figsize=(12, 6))
    plt.title("Generator and Discriminator Loss (Train vs Validation)")
    
    # Plot Train Losses (solid lines)
    plt.plot(train_losses_G, label="Train G Loss", color="blue", linestyle="-")
    plt.plot(train_losses_D, label="Train D Loss", color="orange", linestyle="-")
    
    # Plot Validation Losses (dashed lines)
    plt.plot(val_losses_G, label="Val G Loss", color="blue", linestyle="--")
    plt.plot(val_losses_D, label="Val D Loss", color="orange", linestyle="--")
    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
            
    return generator, discriminator


# ---------------------------------------------------------
# VISUALIZATION UTILITY
# ---------------------------------------------------------
def visualize_prediction(generator, dataloader, device):
    """
    Takes a single batch from the dataloader, generates a prediction using the trained model,
    and plots the Input Map, Generated Image, and Real Image side-by-side.
    
    Args:
        generator (nn.Module): The trained generator network.
        dataloader (DataLoader): DataLoader to draw the sample from (usually validation or test set).
        device (torch.device): The computation device.
    """
    print("Visualizing model prediction on a sample...")
    generator.eval() # Ensure dropout/batchnorm layers are in evaluation mode
    
    with torch.no_grad():
        # Get one batch of data
        input_map, real_image = next(iter(dataloader))
        input_map = input_map.to(device)
        
        # Generate the fake image
        generated_image = generator(input_map)
        
        # Unnormalize and move tensors to CPU for matplotlib compatibility
        # We grab the first image in the batch [0]
        map_viz = unnormalize(input_map[0].cpu()).permute(1, 2, 0)
        gen_viz = unnormalize(generated_image[0].cpu()).permute(1, 2, 0)
        real_viz = unnormalize(real_image[0].cpu()).permute(1, 2, 0)
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(map_viz)
        axes[0].set_title("Input Map")
        axes[0].axis("off")
        
        axes[1].imshow(gen_viz)
        axes[1].set_title("Generated Image")
        axes[1].axis("off")
        
        axes[2].imshow(real_viz)
        axes[2].set_title("Real Image (Target)")
        axes[2].axis("off")
        
        plt.show()
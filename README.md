# Pix2Pix Satellite Image Generation (Maps)

This repository contains a PyTorch implementation of the Pix2Pix architecture for Image-to-Image translation, specifically converting geographic label maps into realistic satellite images using the Maps dataset.

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/danigonzaalez/cv2-maps-pix2pix.git
cd cv2-maps-pix2pix
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained models:

Download the models from the [Releases page](https://github.com/danigonzaalez/cv2-maps-pix2pix/releases) and place them inside a `models/` folder within the cloned repository.

## Running the Demo

You can test the downloaded models using any of the provided sample images. For instance, run the following command from the terminal:

```bash
python demo.py --image maps/image_001.jpg --model models/epoch_200_G_lambda_100.pth --output results/
```

This will run inference on the sample map and save a comparative plot (Input vs Generated vs Real) inside the results/ directory using the indicated model.

*Note: You can replace the input image and the model used for the inference.*

## Help and Options

If you are unsure about the required arguments or want to see all available options for the demo script, you can display the help menu at any time by running:

```bash
python demo.py --help
```
## Train and Evaluate Models

Two notebooks are provided in the `training_and_evaluation` folder:

* **`training.ipynb`**: Used to train the models. Running this notebook will automatically save the trained models into a new folder for later use.
* **`Benchmark.ipynb`**: Used to evaluate the saved models.

Before running either notebook, it is highly recommended to replicate the original training environment using the provided `training_environment.yml` file.
Also the notebooks should be located at the same folder path than the `utils.py` and the `/maps folder`.
   

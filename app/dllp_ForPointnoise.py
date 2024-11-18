from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from DLLPfornoise.utils import normalize, variable_to_cv2_image
import cv2
import torch.nn as nn
from torch.autograd import Variable

# Define app and project directories
app_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(app_dir, '..'))

# Add project root to sys.path for module imports
sys.path.append(project_root)

app = Flask(__name__,
            template_folder=os.path.join(project_root, 'templates'),
            static_folder=os.path.join(project_root, 'static'))

# Set the upload and output folders
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
OUTPUT_FOLDER = os.path.join(project_root, 'static')  # To serve output images via URL
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


from unet.unetmodel import UNet
from DLLPfornoise.models import DLLPfornoise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the UNet model
def load_unet_model():
    model = UNet()
    model_path = os.path.join(project_root, 'unet', 'unet_denoise.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model.to(device)

# Define expand_image_size function
def expand_image_size(imorig):
    """Expand the image size if it is odd."""
    expanded_h = expanded_w = False
    if imorig.shape[2] % 2 == 1:
        expanded_h = True
        imorig = np.concatenate((imorig, imorig[:, :, -1:, :]), axis=2)

    if imorig.shape[3] % 2 == 1:
        expanded_w = True
        imorig = np.concatenate((imorig, imorig[:, :, :, -1:]), axis=3)

    return imorig, expanded_h, expanded_w

# Load models
unet_model = load_unet_model()

# Define routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle the uploaded image
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = file.filename
            input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_image_path)

            # Process the image
            output_filename, original_filename = process_image(input_image_path, filename)

            return redirect(url_for('display_image', original_filename=original_filename, denoised_filename=output_filename))
    return render_template('index.html')

def process_image(input_image_path, filename):
    in_ch = 1
    model_fn = os.path.join(project_root, 'DLLPnet', 'models', 'net_v1.pth')

    imorig = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    imorig = np.expand_dims(imorig, 0)
    imorig = np.expand_dims(imorig, 0)

    # Expand image size if needed
    imorig, expanded_h, expanded_w = expand_image_size(imorig)
    imorig = normalize(imorig)
    imorig = torch.Tensor(imorig)

    # Load model
    net = DLLPfornoise(num_input_channels=in_ch)
    state_dict = torch.load(model_fn)
    model = nn.DataParallel(net).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Set data type
    dtype = torch.FloatTensor if device == torch.device('cpu') else torch.cuda.FloatTensor

    imnoisy = imorig.clone()

    # Denoising process
    with torch.no_grad():
        imorig, imnoisy = Variable(imorig.type(dtype)), Variable(imnoisy.type(dtype))
        nsigma = Variable(torch.FloatTensor([60/255.]).type(dtype))

        im_noise_estim = model(imnoisy, nsigma)
        outim = torch.clamp(imnoisy - im_noise_estim, 0., 1.)

    # Crop back if expanded
    if expanded_h:
        outim = outim[:, :, :-1, :]

    if expanded_w:
        outim = outim[:, :, :, :-1]

    # Convert and save output image
    output_image_cv2 = variable_to_cv2_image(outim)
    output_filename = 'output_' + filename
    output_image_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_image_path, output_image_cv2)

    return output_filename, filename

@app.route('/display')
def display_image():
    original_filename = request.args.get('original_filename')
    denoised_filename = request.args.get('denoised_filename')
    return render_template('display.html', original_filename=original_filename, denoised_filename=denoised_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

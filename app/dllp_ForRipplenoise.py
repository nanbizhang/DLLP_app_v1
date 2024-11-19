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

# 定义应用程序目录和项目根目录
app_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.abspath(os.path.join(app_dir, '..'))

# 将项目根目录添加到 sys.path，以便导入模块
sys.path.append(project_root)

app = Flask(__name__,
            template_folder=os.path.join(project_root, 'templates'),
            static_folder=os.path.join(project_root, 'static'))

# 设置上传和输出文件夹
UPLOAD_FOLDER = os.path.join(project_root, 'uploads')
OUTPUT_FOLDER = os.path.join(project_root, 'static')  # 用于通过URL提供输出图像
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传和输出目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


from unet.unetmodel import UNet
from DLLPfornoise.models import DLLPfornoise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 UNet 模型
def load_unet_model():
    model = UNet()
    model_path = os.path.join(project_root, 'unet', 'unet_denoise.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

# 加载 FFDNet 模型
def load_DLLP_model():
    in_ch = 3
    net = DLLPfornoise(num_input_channels=in_ch)
    model_fn = os.path.join(project_root, 'DLLPfornoise', 'models', 'net_rgb.pth')
    state_dict = torch.load(model_fn, map_location=device)
    model = nn.DataParallel(net).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 定义 expand_image_size 函数
def expand_image_size(imorig):
    """如果图像尺寸为奇数，则扩展图像尺寸。"""
    expanded_h = expanded_w = False
    if imorig.shape[2] % 2 == 1:
        expanded_h = True
        imorig = np.concatenate((imorig, imorig[:, :, -1:, :]), axis=2)

    if imorig.shape[3] % 2 == 1:
        expanded_w = True
        imorig = np.concatenate((imorig, imorig[:, :, :, -1:]), axis=3)

    return imorig, expanded_h, expanded_w

# 加载模型
unet_model = load_unet_model()
DLLP_model = load_DLLP_model()

# 定义路由
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 处理上传的图像
        if 'file' not in request.files:
            return '未找到文件部分'
        file = request.files['file']
        if file.filename == '':
            return '未选择文件'
        if file:
            filename = file.filename
            input_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(input_image_path)

            # 处理图像
            output_filename, original_filename = process_image(input_image_path, filename)

            return redirect(url_for('display_image', original_filename=original_filename, denoised_filename=output_filename))
    return render_template('index.html')

def process_image(input_image_path, filename):

    # 使用 UNet 处理图像
    # 加载并预处理图像
    noisy_image = Image.open(input_image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    noisy_image_tensor = transform(noisy_image).unsqueeze(0).to(device)

    # 使用 UNet 进行去噪
    with torch.no_grad():
        denoised_image_tensor = unet_model(noisy_image_tensor)
    denoised_image = denoised_image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)

    # 将去噪后的图像转换为 PIL 格式并保存为临时文件
    temp_gray_path = os.path.join(app_dir, 'temp_gray.png')
    denoised_image_pil = Image.fromarray((denoised_image * 255).astype('uint8'))
    denoised_image_pil.save(temp_gray_path)


    in_ch = 3
    imorig = cv2.imread(temp_gray_path)
    imorig = cv2.cvtColor(imorig, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    imorig = np.expand_dims(imorig, 0)

    # 扩展图像尺寸（如有必要）
    imorig, expanded_h, expanded_w = expand_image_size(imorig)
    imorig = normalize(imorig)
    imorig = torch.Tensor(imorig)

    model = DLLP_model  # 使用已加载的模型

    # 设置数据类型
    dtype = torch.FloatTensor if device == torch.device('cpu') else torch.cuda.FloatTensor

    imnoisy = imorig.clone()

    # 去噪过程
    with torch.no_grad():
        imorig, imnoisy = Variable(imorig.type(dtype)).to(device), Variable(imnoisy.type(dtype)).to(device)
        nsigma = Variable(torch.FloatTensor([60/255.]).type(dtype)).to(device)

        im_noise_estim = model(imnoisy, nsigma)
        outim = torch.clamp(imnoisy - im_noise_estim, 0., 1.)

    # 如果图像尺寸被扩展，裁剪回原始尺寸
    if expanded_h:
        outim = outim[:, :, :-1, :]

    if expanded_w:
        outim = outim[:, :, :, :-1]

    # 使用 variable_to_cv2_image 函数转换并保存图像
    output_image_cv2 = variable_to_cv2_image(outim)
    output_filename = 'output_' + filename
    output_image_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_image_path, output_image_cv2)

    # 删除临时的灰度图像文件
    os.remove(temp_gray_path)

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

import os
import sys
import cv2
import torch
import numpy as np
from flask import Flask, request, render_template, send_from_directory, jsonify
from tqdm import tqdm

# 获取当前文件的目录，即 app 文件夹路径
app_dir = os.path.abspath(os.path.dirname(__file__))

# 获取 DLLPnet 文件夹的路径
dllpnet_dir = os.path.abspath(os.path.join(app_dir, '..', 'DLLPnet'))

# 将 DLLPnet 文件夹添加到 sys.path
sys.path.append(dllpnet_dir)

from Trainer import Model  # 导入您已有的模型
import config as cfg
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 Flask
app = Flask(__name__,
            template_folder=os.path.join(dllpnet_dir, 'templates'),
            static_folder=os.path.join(dllpnet_dir, 'static'))

UPLOAD_FOLDER = os.path.join(dllpnet_dir, 'static', 'uploads')
PREDICT_FOLDER = os.path.join(dllpnet_dir, 'static', 'predictions')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)

# 初始化模型
cfg.MODEL_CONFIG['LOGNAME'] = 'ours'
cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(F=32, depth=[2, 2, 2, 4, 4])
model = Model()
model.load_model()
model.eval()

# 将图像转换为 Tensor
def img_to_tensor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (torch.tensor(img.transpose(2, 0, 1)).float().to(device) / 255.).unsqueeze(0)

# 保存预测结果
def save_predicted_image(mid, target):
    mid_numpy = mid.squeeze().cpu().numpy().transpose(1, 2, 0)
    mid_numpy = (np.clip(mid_numpy, 0, 1) * 255).astype(np.uint8)
    mid_numpy = cv2.cvtColor(mid_numpy, cv2.COLOR_RGB2BGR)
    path = os.path.join(PREDICT_FOLDER, f'predicted_img{target}.jpg')
    cv2.imwrite(path, mid_numpy)
    return path

# 处理上传的图片并生成预测
@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('images')
    if len(files) != 2:
        return jsonify({'error': '请上传两张图片'})

    # 保存上传的图片
    img_paths = []
    for idx, file in enumerate(files):
        path = os.path.join(UPLOAD_FOLDER, f'img{idx}.jpg')
        file.save(path)
        img = cv2.imread(path)
        if img is None:
            return jsonify({'error': f'无法加载图片 img{idx}.jpg'})
        img_paths.append(img)

    # 转换图片为张量
    I0 = img_to_tensor(img_paths[0])
    I4 = img_to_tensor(img_paths[1])

    # 预测中间帧并保存
    I2 = model.inference(I0, I4, TTA=True)[0].unsqueeze(0)
    path2 = save_predicted_image(I2, 2)

    I1 = model.inference(I0, I2, TTA=True)[0].unsqueeze(0)
    path1 = save_predicted_image(I1, 1)

    I3 = model.inference(I2, I4, TTA=True)[0].unsqueeze(0)
    path3 = save_predicted_image(I3, 3)

    # 返回相对于静态文件夹的路径
    return jsonify({
        'img1': f'static/predictions/predicted_img1.jpg',
        'img2': f'static/predictions/predicted_img2.jpg',
        'img3': f'static/predictions/predicted_img3.jpg'
    })

# 静态资源路由
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(os.path.join(dllpnet_dir, 'static'), path)

# 前端页面
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

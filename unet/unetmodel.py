import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# 加载数据集
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert("RGB")
        if img is not None:
            images.append(img)
    return images



# 定义UNet模型中的基本卷积块
class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

# 定义UNet模型
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        # 编码器部分
        self.enc1 = UNetConvBlock(in_channels, 64)
        self.enc2 = UNetConvBlock(64, 128)
        self.enc3 = UNetConvBlock(128, 256)
        self.enc4 = UNetConvBlock(256, 512)

        # 最大池化层
        self.pool = nn.MaxPool2d(2)

        # 中间层
        self.middle = UNetConvBlock(512, 1024)

        # 解码器部分
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = UNetConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetConvBlock(128, 64)

        # 输出层
        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器前向传播
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # 中间层
        middle = self.middle(self.pool(enc4))

        # 解码器前向传播
        dec4 = self.up4(middle)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.up3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        # 输出层
        output = self.output_layer(dec1)
        return output

# 自定义数据集类
class NoisyDataset(Dataset):
    def __init__(self, noisy_images, clean_images, transform=None):
        self.noisy_images = noisy_images
        self.clean_images = clean_images
        self.transform = transform

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy = self.noisy_images[idx]
        clean = self.clean_images[idx]
        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)
        return noisy, clean

# 训练UNet模型
def train_unet(model, train_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), 'unet_denoise.pth')
    print("模型已保存为 unet_denoise.pth")

# 加载模型并对单张图片进行去噪
def denoise_image(model, noisy_image_path, device='cpu'):
    model.load_state_dict(torch.load('unet_denoise.pth'))
    model.to(device)
    model.eval()

    # 读取并预处理图像
    noisy_image = Image.open(noisy_image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    noisy_image_tensor = transform(noisy_image).unsqueeze(0).to(device)

    # 去噪
    with torch.no_grad():
        denoised_image_tensor = model(noisy_image_tensor)
    denoised_image = denoised_image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)

    # 显示去噪前后对比
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Noisy Image")
    plt.imshow(noisy_image)
    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_image)
    plt.show()

# 主程序
if __name__ == "__main__":
    # 数据加载
    # 替换成你存储干净图像和有噪声图像的文件夹路径
    clean_images_folder = r'C:\Users\Administrator\Desktop\uuunet\colon_v2\all-clean'
    noisy_images_folder = r'C:\Users\Administrator\Desktop\uuunet\colon_v2\all-noisy'

    # 加载干净和有噪声的图像
    clean_images = load_images_from_folder(clean_images_folder)
    noisy_images = load_images_from_folder(noisy_images_folder)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = NoisyDataset(noisy_images, clean_images, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # 初始化模型、损失函数和优化器
    model = UNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 训练模型
    train_unet(model, train_loader, criterion, optimizer, num_epochs=100, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 去噪单张图片
    test_image_path = r"C:\Users\Administrator\Desktop\uuunet\colon_v2\all-noisy\66.png"
    denoise_image(model, test_image_path, device='cuda' if torch.cuda.is_available() else 'cpu')
 
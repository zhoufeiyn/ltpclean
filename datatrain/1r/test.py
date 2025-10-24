from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode
image_size = 256
img = Image.open("zf_f14_a2_nt1.png")
t = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
    transforms.Pad((0, 16, 0, 16), fill=(107, 140, 255)),  # 上下各加16像素天空蓝
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
x = t(img)
# 反归一化
x = (x * 0.5 + 0.5).clamp(0, 1)
plt.imshow(x.permute(1, 2, 0))
plt.show()

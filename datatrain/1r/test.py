from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode
image_size = 256
img = Image.open("zf_f14_a2_nt1.png")
t = transforms.Compose([
        # 按宽度等比例缩放，保持比例不变形
        transforms.Lambda(
            lambda img: transforms.functional.resize(
                img,
                size=(int(img.height * (256 / img.width)), 256),  # → 256×192
                interpolation=InterpolationMode.NEAREST
            )
        ),
        # 居中补上下边，使高宽都为256
        transforms.Lambda(
            lambda img: transforms.functional.pad(
                img,
                padding=(0, (256 - img.height) // 2, 0, 256 - img.height - (256 - img.height) // 2),
                fill=(107, 140, 255)
            )
        ),
        # 转Tensor并归一化到[-1,1]
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
x = t(img)
print(x.shape)
# 反归一化
x = (x * 0.5 + 0.5).clamp(0, 1)
plt.imshow(x.permute(1, 2, 0))
plt.show()

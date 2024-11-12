import torch
import torch.nn as nn
from model import efficientnetv2_m as create_model
from resnet import resnet50


class ModelWithFPN(nn.Module):

    # [torch.Size([1, 48, 128, 128]), torch.Size([1, 80, 64, 64]), torch.Size([1, 160, 32, 32]), torch.Size([1, 304, 16, 16])]
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.model = create_model(num_classes=num_classes)
        # self.model = resnet50(num_classes=num_classes)
        # FIXME 直接使用倍率放大，对于某些尺寸的图片 会导致出错， 目前512, 1024 可以， 600报错
        self.conv_up_4 = nn.Sequential(nn.Conv2d(304, 160, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(num_features=160, eps=1e-3, momentum=0.1),
                                       nn.Upsample(scale_factor=2, mode='bilinear'))  # 160 32 32
        self.conv_up_3 = nn.Sequential(nn.Conv2d(160, 80, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(num_features=80, eps=1e-3, momentum=0.1),
                                       nn.Upsample(scale_factor=2, mode='bilinear'))  # 80 64 64
        self.conv_up_2 = nn.Sequential(nn.Conv2d(80, 48, kernel_size=1, bias=False),
                                       nn.BatchNorm2d(num_features=48, eps=1e-3, momentum=0.1),
                                       nn.Upsample(scale_factor=2, mode='bilinear'))  # 48 128 128


        self.SODnet = nn.Sequential(
            nn.ConvTranspose2d(48, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64, eps=1e-3, momentum=0.1),  # 64 256 256
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32, eps=1e-3, momentum=0.1),  # 32 512 512
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        label, [out1, out2, out3, out4] = self.model(x)
        out3 = out3 + self.conv_up_4(out4)
        out2 = out2 + self.conv_up_3(out3)
        out1 = out1 + self.conv_up_2(out2)
        SOD_out = self.SODnet(out1)
        return label, SOD_out


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModelWithFPN(2).to(device)
    random_input = torch.randn(1, 3, 1024, 1024).to(device)
    print([out.shape for out in model(random_input)[1]])
    print(model)

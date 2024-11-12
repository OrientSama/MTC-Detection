import torch
from torchvision.transforms import transforms
from mode_with_fpn import ModelWithFPN
from PIL import Image
import matplotlib.pyplot as plt

def main():
    img_size = {"s": [300, 384],  # train_size, val_size
                # "m": [384, 480],
                "m": [512, 512],
                "l": [384, 480]}
    num_model = "m"
    num_classes = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    model_weight_path = "/home/ubuntu/PycharmProjects/DeepLearn/Test3_Salient_Region/save_weights/last_model_180_best.pth"
    model = ModelWithFPN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
    m = torch.nn.Sigmoid()
    pic_path = "/home/ubuntu/Dataset/DOTA-Split-mmr/val/images/P0179__1024__824___0.png"
    mask_path = "/home/ubuntu/Dataset/DOTA-Split-mmr/val/mask_images/P0179__1024__824___0.jpg"
    img = Image.open(pic_path)
    mask_img = Image.open(mask_path)
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 3)
    plt.imshow(mask_img)
    img = data_transform(img).to(device).unsqueeze(dim=0)
    pred, mask_output = model(img)
    mask_output = m(mask_output)
    plt.subplot(1, 3, 2)
    mask_output = mask_output.detach().cpu().squeeze().numpy()
    plt.imshow(mask_output)
    print("Pred:", m(pred.detach()))
    plt.show()



if __name__ == '__main__':
    main()
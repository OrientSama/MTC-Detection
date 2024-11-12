import torch
from mode_with_fpn import ModelWithFPN

# model with FPN
model = ModelWithFPN(num_classes=1)
# with open('model.txt', 'w') as f:
#     print(model, file=f)

tmp = torch.randn((1, 3, 512, 512))

out, [out1, out2, out3, out4] = model.model(tmp)

print(out.size())
print(out1.size())
print(out2.size())
print(out3.size())
print(out4.size())
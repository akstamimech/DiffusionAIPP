from base import UNET
from torchviz import make_dot
import torch


model = UNET()

x = torch.randn(1, 3, 64, 64)
t = torch.randint(0, 100, (1,), dtype=torch.long)

y = model(x, t)

dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("unet_graph", format="png")
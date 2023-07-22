import torch
from torch import nn
from torchviz import make_dot
from torch.autograd import Variable

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)

    def forward(self, x):
        x = self.linear(x)

        return x

t = torch.rand(1,1, requires_grad=True)

print(t)
print("t.grad ", t.grad)
t1 = t * 2

t2 = t1 * 3
t3 = t2 * 5
t4 = t3 * 5
t5 = t4 * 9

target = torch.rand(1,1,requires_grad=True)
criterion = nn.MSELoss()



model = Net()

out = model(Variable(t5))
loss = criterion(t2, target)
make_dot(loss, params=dict(model.named_parameters()))

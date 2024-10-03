import pytest
import torch
import torch.nn as nn
import copy

from src.utils.adamw_scaled import AdamWScale

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Function to train the model and return final loss
def train_model(model, optimizer, criterion, inputs, logits, num_epochs=3):
    for _ in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, logits)
        loss.backward()
        optimizer.step()
    return loss.item()

@pytest.mark.parametrize("use_bfloat16", [False, True])
@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("kahan_sum", [False, True])
@pytest.mark.parametrize("weight_decay", [0.0, 0.03])
def test_optimizers(use_bfloat16, foreach, kahan_sum, weight_decay):
    torch.manual_seed(42)
    model1 = SimpleNet().cuda()
    model2 = copy.deepcopy(model1)

    criterion = nn.MSELoss()
    inputs = torch.randn(32, 10).cuda()
    logits = torch.randn(32, 1).cuda()

    if use_bfloat16:
        inputs = inputs.to(torch.bfloat16)
        logits = logits.to(torch.bfloat16)
        model1 = model1.to(torch.bfloat16)
        model2 = model2.to(torch.bfloat16)

    optimizer1 = AdamWScale(model1.parameters(), lr=0.01, foreach=False, kahan_sum=False, weight_decay=weight_decay)
    optimizer2 = AdamWScale(model2.parameters(), lr=0.01, foreach=foreach, kahan_sum=kahan_sum, weight_decay=weight_decay)

    loss1 = train_model(model1, optimizer1, criterion, inputs, logits)
    loss2 = train_model(model2, optimizer2, criterion, inputs, logits)

    assert abs(loss1 - loss2) < 0.1, "Optimizers perform significantly differently"

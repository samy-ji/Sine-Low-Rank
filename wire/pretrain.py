import torch
import torch.nn as nn
import torch.optim as optim
import math

# Initialize matrix A with Kaiming initialization
def topk_linf_loss(output, target, k=1):
    errors = torch.abs(output - target)
    topk_errors, _ = torch.topk(errors.view(-1), k)
    return topk_errors[-1]

# Define the model
class MatrixOptimization(nn.Module):
    def __init__(self,rank_k):
        super(MatrixOptimization, self).__init__()
        # Initialize B and C as learnable parameters
        self.B = nn.Parameter(torch.randn(256, rank_k))
        self.C = nn.Parameter(torch.randn(rank_k, 256))

    def forward(self):
        return torch.matmul(self.B, self.C)


def pretrain(rank_k,w=None):
    # Create the model instance
    #A = torch.nn.init.kaiming_normal_(torch.empty(256, 256), mode='fan_in', nonlinearity='relu')
    A = torch.empty(256,256)
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    model = MatrixOptimization(rank_k)

    # Define the loss function and the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 30000
    for epoch in range(num_epochs):
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model()
        loss1 = nn.MSELoss()(output,A)
        #loss2 = topk_linf_loss(output,A,k=1000)

        loss = loss1
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Print loss every 100 epochs
        if (epoch+1) % 1000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss1: {loss1.item()}')
    return model.B.detach(),model.C.detach()


#torch.save(model.B.detach(), 'results/B_matrix.pt')
#torch.save(model.C.detach(), 'results/C_matrix.pt')
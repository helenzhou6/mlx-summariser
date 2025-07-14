import torch 

# Lora example
class Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(10, 5)
        self.lora = torch.nn.Sequential(
            torch.nn.Linear(10, 2, bias=False),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(2, 5, bias=False)
        )
    
    def forward(self, x):
        proj_x = self.proj(x)
        output_lora = self.lora(x)
        return proj_x + output_lora # residual connection

base = Base()
nums = sum(p.numel() for p in base.parameters())
print('Params: ', nums)
print(base)
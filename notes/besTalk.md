# Preference Optimisation

## PEFT (Parameter-Efficient Fine-Tuning)** 
Class of techniques (esp in large language models) to adapt a pretrained model to a new specific task or domain by tuning only a small subset of its parameters instead of all of them.
- Since otherwise fine tuning all params for each new task is computationally expensive
PEFT modifies or adds a small number of params e.g.:
- **Adapters**: Small bottleneck separate layers inserted inside the pretrained model layers. (or extra function) Only adapter layer weights are trained.
- **LoRa (Low-Rank Adaptation)**: Decomposes weight updates into low-rank matrices and only updates these small matrices. (i.e. injected into existing weight matrices + residual connection)
- **Prefix-tuning**: Adds a small learned vector (prefix) to the input embeddings or hidden states and tunes only that.
- **BitFit**: Fine-tunes only the bias terms of the model.

### LoRa
- Normally take an input, take the hidden state, project input down, then project back up to original embedding size
- This output and the froze pretrained weights are added together (residual connection)
- e.g for the following code - only 3 lines in the Sequential are trained
```python
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
        x = self.proj(x)
        return x + self.lora(x)

base = Base()
nums = sum(p.numel() for p in base.parameters())
print('Params: ', nums)
print(base)
```
- You can apply can apply LoRa to Q/K/V of attention - or just Q.
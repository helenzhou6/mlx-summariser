# Preference Optimisation

## PEFT (Parameter-Efficient Fine-Tuning)

<img width="1095" height="412" alt="Screenshot 2025-07-14 at 14 56 18" src="https://github.com/user-attachments/assets/b007e0b2-e734-4a3a-aba0-41a901e633b5" />

Class of techniques (esp in large language models) to adapt a pretrained model to a new specific task or domain by tuning only a small subset of its parameters instead of all of them.
- Since otherwise fine tuning all params for each new task is computationally expensive
PEFT modifies or adds a small number of params e.g.:
- **Adapters**: Small bottleneck separate layers inserted inside the pretrained model layers. (or extra function) Only adapter layer weights are trained.
- **LoRa (Low-Rank Adaptation)**: Decomposes weight updates into low-rank matrices and only updates these small matrices. (i.e. injected into existing weight matrices + residual connection)
- **Prefix-tuning**: Adds a small learned vector (prefix) to the input embeddings or hidden states and tunes only that.
- **BitFit**: Fine-tunes only the bias terms of the model.

### LoRa
<img width="1098" height="427" alt="Screenshot 2025-07-14 at 14 56 33" src="https://github.com/user-attachments/assets/4a024461-e0df-407a-9648-8c51ce2493a8" />
<img width="942" height="424" alt="Screenshot 2025-07-14 at 14 56 49" src="https://github.com/user-attachments/assets/124c70cc-71ba-4e72-bdb0-efc1a458ca28" />

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


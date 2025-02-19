<h1 align="center">
    <p> Efficient Learning with Sine-Activated Low-Rank Matrices </p>
</h1> 

[[`Paper`](https://arxiv.org/abs/2403.19243)] [[`Website`](https://github.com/samy-ji/Sine-Low-Rank/)] 

The official PyTorch implementation of [**Efficient Learning with Sine-Activated Low-Rank Matrices**](https://arxiv.org/abs/2403.19243) 

Sine-LoRA applies a sine function to improve the spectrum of low-rank decompositions for fine-tuning. 

| <img src="svd_1.png" alt="Figure 1" width="250"> | <img src="svd_2.png" alt="Figure 2" width="250"> |
|--------------------------------------------------|--------------------------------------------------|

Applying a sine nonlinearity to a low-rank matrix  $\phi(\mathbf{x}) = \sin(\omega \mathbf{U} \mathbf{V}^{T})$ has improved spectral properties. $\omega$ controls the spectrum smoothness.

## Sine LoRA

```
## LoRA forward pass
  def forward(self, x: torch.Tensor):
    result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    result += ((self.lora_dropout(x.to(self.lora_A.weight.dtype)) @ self.lora_A.weight.T) @ self.lora_B.weight.T) * self.scaling
    return result

## Sine LoRA forward pass
  def forward(self, x: torch.Tensor):
    result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
    result += ((self.lora_dropout(x.to(self.lora_A.weight.dtype))) @ torch.sin(self.freq * self.lora_A.weight.T @ self.lora_B.weight.T))/self.s * self.scaling
    return result
```

**Performance and parameter count of the LLaMA 3-8B model fine-tuned using the LoRA and sine-LoRA methods**

| **Method**              | **Params** | **BoolQ** | **PIQA** | **SIQA** | **HS**  | **WG**  | **ARC-e** | **ARC-c** | **OBQA** | **Avg.** |
|--------------------------|------------|-----------|----------|----------|---------|---------|-----------|-----------|----------|----------|
| LoRA<sub>k=4</sub>      | 7.1M       | 73.58     | 86.29    | 79.99    | 94.92   | 79.95    | 63.91      | 88.7       | 83.0      | 80.04    |
| Sine LoRA<sub>k=4</sub> | 7.1M       | 72.69     | 87.38    | 79.32    | 94.39   | 85.32    | 75.01      | 88.64      | 86.2      | **83.61** |
| LoRA<sub>k=8</sub>      | 14.2M      | 72.97     | 87.43    | 78.81    | 72.18   | 85.80    | 77.47      | 88.38      | 83.20     | 80.79   |
| Sine LoRA<sub>k=8</sub> | 14.2M      | 73.42     | 86.51    | 80.3     | 94.16   | 85.87    | 76.36      | 88.05      | 84.6      | **83.66** |
| LoRA<sub>k=16</sub>     | 28.3M      | 73.57     | 85.58    | 79.27    | 93.97   | 85.71    | 75.42      | 86.44      | 83.20     | 82.90    |
| Sine LoRA<sub>k=16</sub>| 28.3M      | 73.7      | 87.65    | 80.76    | 94.93   | 84.45    | 79.1       | 89.77      | 84.4      | **84.35** |
| LoRA<sub>k=32</sub>     | 56.6M      | 70.64     | 86.13    | 78.25    | 91.48   | 83.19    | 69.71      | 85.73      | 81.40     | 80.82    |
| Sine LoRA<sub>k=32</sub>| 56.6M      | 72.42     | 86.51    | 79.78    | 93.96   | 85.16    | 78.07      | 87.58      | 85.0      | **83.56** |

## Sine DoRA

```
## DoRA forward pass
def forward(self, x: torch.Tensor):
    base_result = F.linear(x, transpose(self.weight, self.fan_in_fan_out))
    dropout_x = self.lora_dropout(x)

    new_weight_v = self.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling
    norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
    result = base_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
    result += ( norm_scale * (self.lora_B(self.lora_A(dropout_x.to(self.lora_A.weight.dtype))))) * self.scaling
    if not self.bias is None:
      result += self.bias.view(1, -1).expand_as(result)
    return result

## Sine DoRA forward pass

def forward(self, x: torch.Tensor):
    base_result = F.linear(x, transpose(self.weight, self.fan_in_fan_out))
    dropout_x = self.lora_dropout(x)

    new_weight_v = self.weight + torch.sin(self.freq*(self.lora_B.weight @ self.lora_A.weight))/self.s * self.scaling 
    norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
    result = base_result + (norm_scale-1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))
    result += (norm_scale * torch.sin(self.freq*(self.lora_B(self.lora_A(dropout_x.to(self.lora_A.weight.dtype))))/self.s)) * self.scaling
    if not self.bias is None:
      result += self.bias.view(1, -1).expand_as(result)
    return result

```


**Performance and parameter count of the LLaMA 3-8B model fine-tuned using the DoRA and sine-DoRA methods**

| **Method**              | **Params** | **BoolQ** | **PIQA** | **SIQA** | **HS**  | **WG**  | **ARC-e** | **ARC-c** | **OBQA** | **Avg.** |
|--------------------------|------------|-----------|----------|----------|---------|---------|-----------|-----------|----------|----------|
| DoRA<sub>k=8</sub>      | 14.9M      | 73.2      | 87.7     | 79.9     | 94.7    | 84.5    | 89.3      | 78.0      | 83.2     | 83.8     | 
| Sine DoRA<sub>k=8</sub> | 14.9M      | 73.9  | 89.0 | 81.0 | 95.3| 86.1 | 90.1  | 79.0  | 87.0 | **85.2** |        
| DoRA<sub>k=16</sub>    | 29.1M      | 74.5      | 88.8     | 80.3     | 95.5| 84.7    | 90.1  | 79.1      | 87.2     | 85.0     | 
| Sine DoRA<sub>k=16</sub>| 29.1M      | 75.1  | 89.0 | 81.0 | 95.3    | 86.1| 90.0      | 79.3  | 86.2 | **85.3** |      
| DoRA<sub>k=32</sub>    | 57.4M      | 74.6      | 89.3 | 79.9     | 95.5    | 85.6    | 90.5  | 80.4  | 85.8 | 85.2     | 
| Sine DoRA<sub>k=32</sub>| 57.4M      |  75.8    |   89.3   |   80.3   |   95.9 |  86.1 | 90.2      | 79.4      | 85.4     | **85.3** |       



[1] Liu et al. DoRA: Weight-Decomposed Low-Rank Adaptation. ICML 2024

## Contact
Yiping Ji [yiping.ji@adelaide.edu.au](yiping.ji@adelaide.edu.au) or Hemanth Saratchandran [hemanth.saratchandran@adelaide.edu.au](hemanth.saratchandran@adelaide.edu.au).

## Citation
If you find Sine-LoRA useful, please consider giving a star and citation:

```bibtex
@article{ji2024sine,
  title={Efficient Learning with Sine-Activated Low-Rank Matrices},
  author={Ji, Yiping and Saratchandran, Hemanth and Gordon, Cameron and Zhang, Zeyu and Lucey, Simon},
  journal={arXiv preprint arXiv:2403.19243},
  year={2024}
}
```

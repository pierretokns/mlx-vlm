import math
from typing import Union

import mlx.core as mx
import mlx.nn as nn


class LoRaLayer(nn.Module):
    def __init__(
        self,
        linear: Union[nn.Linear, nn.QuantizedLinear],
        rank: int,
        alpha: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.original_layer = linear

        self.dropout = nn.Dropout(p=dropout)

        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits

        std_dev = 1 / math.sqrt(rank)

        self.A = mx.random.uniform(
            low=-std_dev,
            high=std_dev,
            shape=(input_dims, rank),
        )
        self.B = mx.zeros((rank, output_dims))
        self.alpha = alpha

    def __call__(self, x):
        y = self.original_layer(x)
        lora_update = (self.dropout(x) @ self.A) @ self.B
        return y + (self.alpha * lora_update).astype(x.dtype)

    def fuse(self, dequantize: bool = False):
        """Fuse LoRA weights into the original layer.

        Args:
            dequantize: If True, return a regular Linear even if original was quantized.

        Returns:
            nn.Linear or nn.QuantizedLinear with LoRA weights merged in.
        """
        linear = self.original_layer
        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)

        if is_quantized:
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                group_size=linear.group_size,
                bits=linear.bits,
            )

        output_dims, input_dims = weight.shape
        bias = "bias" in linear
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Compute fused weight: W' = W + alpha * (A @ B).T
        # A is (input_dims, rank), B is (rank, output_dims)
        # Weight is (output_dims, input_dims), so delta needs transpose
        delta = (self.alpha * (self.A @ self.B)).T.astype(weight.dtype)
        fused_linear.weight = weight + delta

        if bias:
            fused_linear.bias = linear["bias"]

        if is_quantized and not dequantize:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
            )

        return fused_linear


def replace_lora_with_linear(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, LoRaLayer):
            # Compute the final merged weight
            lora_update = layer.alpha * (layer.A @ layer.B)
            updated_weight = layer.original_layer.weight + lora_update
            use_bias = layer.original_layer.bias is not None

            updated_bias = layer.original_layer.bias

            # Create a new Linear layer with the updated parameters
            new_linear_layer = nn.Linear(
                updated_weight.size(1), updated_weight.size(0), bias=use_bias
            )

            new_linear_layer.weight = updated_weight

            if use_bias:
                new_linear_layer.bias = updated_bias

            if isinstance(layer.original_layer, nn.QuantizedLinear):
                new_linear_layer = nn.QuantizedLinear.from_linear(
                    new_linear_layer,
                    new_linear_layer.group_size,
                    new_linear_layer.bits,
                )

            # Replace the LoRaLayer with the new Linear layer in the model
            model.layers[i] = new_linear_layer

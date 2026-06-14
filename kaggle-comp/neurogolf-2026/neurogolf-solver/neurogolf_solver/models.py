from __future__ import annotations

from pathlib import Path
from typing import Callable

import onnx
from onnx import TensorProto, helper

BATCH_SIZE = 1
CHANNELS = 10
HEIGHT = 30
WIDTH = 30
GRID_SHAPE = [BATCH_SIZE, CHANNELS, HEIGHT, WIDTH]
IR_VERSION = 10
OPSET_IMPORTS = [helper.make_opsetid("", 10)]
WeightFn = Callable[[int, int, tuple[int, int]], float]


def single_layer_conv2d_network(weight_fn: WeightFn, kernel_size: int = 1) -> onnx.ModelProto:
    kernel_offsets = range(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    weight_shape = [CHANNELS, CHANNELS, kernel_size, kernel_size]
    weights = [
        weight_fn(channel_out, channel_in, (row_offset, col_offset))
        for channel_out in range(CHANNELS)
        for channel_in in range(CHANNELS)
        for row_offset in kernel_offsets
        for col_offset in kernel_offsets
    ]

    network_input = helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)
    network_output = helper.make_tensor_value_info("output", TensorProto.FLOAT, GRID_SHAPE)
    weight_tensor = helper.make_tensor("W", TensorProto.FLOAT, weight_shape, weights)
    conv_node = helper.make_node(
        "Conv",
        ["input", "W"],
        ["output"],
        kernel_shape=[kernel_size, kernel_size],
        pads=[kernel_size // 2] * 4,
    )
    graph = helper.make_graph([conv_node], "graph", [network_input], [network_output], [weight_tensor])
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def identity_network() -> onnx.ModelProto:
    def keep_same_channel(channel_out: int, channel_in: int, kernel_coord: tuple[int, int]) -> float:
        return 1.0 if channel_out == channel_in and kernel_coord == (0, 0) else 0.0

    return single_layer_conv2d_network(keep_same_channel, kernel_size=1)


def save_model(model: onnx.ModelProto, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path

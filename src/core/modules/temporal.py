from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WaveNetLayer(nn.Module):
    def __init__(
        self,
        num_channels: int,
        kernel_size: int,
        dilation: int,
        drop: float = 0.25,
        leaky: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.leaky = leaky
        self.dilated_conv = nn.Conv1d(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.dilation,
        )
        self.conv_1x1 = nn.Conv1d(
            in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=1
        )
        self.drop = nn.Dropout(drop)

        if self.leaky:
            self.non_lin_func = F.leaky_relu
        else:
            self.non_lin_func = F.relu

    def apply_non_lin(self, y: Tensor) -> Tensor:
        return self.non_lin_func(y)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [B x num_channels x T]
        :return: [B x num_channels x T]
        """
        y = self.dilated_conv.forward(x)
        y = self.apply_non_lin(y)  # non-linearity
        y = self.conv_1x1.forward(y)
        y = self.drop.forward(y)  # dropout
        y += x  # residual connection
        return y


class NoFt(nn.Module):
    def __init__(self, in_chnnels: int, out_dims: int, kernel_size: int = 1):
        super().__init__()
        self.in_chnnels = in_chnnels
        self.out_dims = out_dims
        self.kernel_size = kernel_size

        self.last_conv = nn.Conv1d(
            in_channels=self.in_chnnels,
            out_channels=self.out_dims,
            kernel_size=self.kernel_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [B x in_channels x T]
        :return: [B x out_dims x T]
        """
        return self.last_conv.forward(x)


class WaveNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stages: List[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
        out_dims: int = 64,
        kernel_size: int = 3,
        pooling=True,
        pooling_layers: Iterable[int] = (1, 2, 4, 8),
        pooling_type: str = "max",  # could be "max" and "sum"
        dropout_rate=0.25,
        leaky=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_stages = len(stages)
        self.stages = stages
        self.out_dims = out_dims
        self.kernel_size = kernel_size
        self.layers = []
        self.pooling = pooling
        self.pooling_type = pooling_type
        self.pooling_layers = pooling_layers
        self.dropout_rate = dropout_rate
        self.leaky = leaky

        if self.leaky:
            self.non_lin_fun = F.leaky_relu
        else:
            self.non_lin_fun = F.relu

        self.first_conv = nn.Conv1d(
            in_channels=self.in_channels, out_channels=self.out_dims, kernel_size=1
        )

        self.last_conv = nn.Conv1d(
            in_channels=self.out_dims, out_channels=self.out_dims, kernel_size=1
        )

        for i in range(self.num_stages):
            stage = self.stages[i]
            layer = WaveNetLayer(
                self.out_dims,
                kernel_size=self.kernel_size,
                dilation=stage,
                drop=self.dropout_rate,
                leaky=self.leaky,
            )
            self.layers.append(layer)
            self.add_module("l_{}".format(i), layer)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [B x in_channels x T]
        :return: [B x out_dims x T]
        """
        x = self.non_lin_fun(self.first_conv.forward(x))

        for i, l in enumerate(self.layers):
            x = l.forward(x)
            if i in self.pooling_layers and self.pooling:
                if self.pooling_type == "max":
                    x = F.max_pool1d(x, kernel_size=2)
                else:
                    x = F.avg_pool1d(x, kernel_size=2)
                    x = x * 2

        x = self.non_lin_fun(x)
        x = self.last_conv.forward(x)

        return x


class MSTCNPPFirstStage(nn.Module):
    def __init__(
        self, num_layers, num_f_maps, input_dim, output_dim, pooling_layers=(1, 2, 4, 8)
    ):
        super().__init__()
        self.num_layers = num_layers
        self.conv_1x1_in = nn.Conv1d(input_dim, num_f_maps, 1)
        self.conv_dilated_1 = nn.ModuleList(
            (
                nn.Conv1d(
                    num_f_maps,
                    num_f_maps,
                    3,
                    padding=2 ** (num_layers - 1 - i),
                    dilation=2 ** (num_layers - 1 - i),
                )
                for i in range(num_layers)
            )
        )
        self.conv_dilated_2 = nn.ModuleList(
            (
                nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i)
                for i in range(num_layers)
            )
        )
        self.conv_fusion = nn.ModuleList(
            (nn.Conv1d(2 * num_f_maps, num_f_maps, 1) for i in range(num_layers))
        )
        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, output_dim, 1)

        self.pooling_layers = pooling_layers

    def forward(self, x):
        """
        :param x: [B x in_channels x T]
        :return: [B x out_dims x T]
        """
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](
                torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1)
            )
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

            if i in self.pooling_layers:
                f = F.max_pool1d(f, kernel_size=2)

        out = self.conv_out(f)

        return out


if __name__ == "__main__":
    inp = torch.rand((1, 64, 1024))

    n1 = WaveNetBlock(
        in_channels=64,
        stages=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        out_dims=64,
        pooling_layers=[1, 2],
    )

    n2 = MSTCNPPFirstStage(
        num_layers=11,
        num_f_maps=64,
        input_dim=64,
        output_dim=64,
        pooling_layers=[1, 2],
    )

    n3 = NoFt(
        in_chnnels=64,
        out_dims=64,
        kernel_size=1
    )

    print(inp.shape)

    o1 = n1.forward(inp)
    o2 = n2.forward(inp)
    o3 = n3.forward(inp)

    print(o1.shape, o2.shape, o3.shape)

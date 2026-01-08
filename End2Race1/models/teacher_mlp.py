import torch
import torch.nn as nn
from typing import Iterable, List


def _build_teacher_mlp(in_dim: int, hidden_dims: Iterable[int], out_dim: int, dropout: float, activation: str) -> nn.Sequential:
    layers: List[nn.Module] = []
    activation_layer = nn.ReLU if activation.lower() == "relu" else nn.GELU

    dims = [in_dim] + list(hidden_dims) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TeacherMLP(nn.Module):
    """
    End2Race의 control head/decision MLP 형태를 참고한 Teacher 전용 MLP.
    encoder feature를 받아 최종 제어 출력을 생성한다.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        out_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        self.teacher_head = _build_teacher_mlp(in_dim, hidden_dims, out_dim, dropout, activation)

    def forward(self, teacher_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            teacher_features: [B, D] encoder에서 나온 특징.
        Returns:
            teacher_outputs: 제어 출력 (steer, throttle 등).
        """
        teacher_outputs = self.teacher_head(teacher_features)
        return teacher_outputs


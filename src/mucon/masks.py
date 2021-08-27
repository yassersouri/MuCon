import torch
from scipy.signal import gaussian
from torch import Tensor
from torch.nn.functional import affine_grid, grid_sample, softmax


# noinspection PyPep8Naming
def project_lengths_softmax(T: int, L: Tensor) -> Tensor:
    """

    :param T: 1:int
    :param L: [M]:float
    :return: [M]:float
    """
    return T * softmax(L, dim=0)


# noinspection PyPep8Naming
def create_masks(
    T: int, L: Tensor, overlap: float = 0.0, template: str = "box"
) -> Tensor:
    """
    Given a set of projected S_{i}s, creates the attentions for weak training.
    :param T: The target size for the masks.
    :param L: [M] the projected lengths.
    :param overlap: how much overlap should the attentions have
    :param template: the kind of template to use. "box", "gaussian"
    :return: [M x T] the attention maps.
    """

    TEMPLATE_WIDTH = 100
    B = L.size(0)

    if template == "gaussian":
        std = TEMPLATE_WIDTH / 5
        template = (
            torch.tensor(gaussian(M=TEMPLATE_WIDTH, std=std))
            .repeat((B, 1))
            .view(B, 1, -1)
        )
        template = template.float().to(L.device)
    elif template == "box":
        template = L.new_ones((B, 1, TEMPLATE_WIDTH))
    elif template == "trapezoid":
        w1 = TEMPLATE_WIDTH / 2
        min_val = 0.5
        template = torch.ones(TEMPLATE_WIDTH)
        template[: int(w1 / 2)] = torch.arange(
            start=min_val, end=1, step=(1 - min_val) / (w1 / 2)
        )
        template[-int(w1 / 2) :] = torch.arange(
            start=1, end=min_val, step=(min_val - 1) / (w1 / 2)
        )
        template = template.repeat((B, 1)).view(B, 1, -1).float().to(L.device)
    else:
        raise NameError(f"Invalid template name ({template})")

    pis = torch.cumsum(L, 0)  # [B]
    pis -= L  # [B]

    L *= 1.0 + 2 * overlap
    pis -= L * (overlap / 2)

    normalized_sis = _normalize_scale(T, L)
    normalized_pis = _normalize_location(T, pis, L)

    params_mat = _create_params_matrix(normalized_sis, normalized_pis)  # [B x 3]
    theta = _create_theta(params_mat)  # [B, 2, 3]

    grid = affine_grid(theta, torch.Size((B, 1, 1, T)))
    out = grid_sample(template.view(B, 1, 1, TEMPLATE_WIDTH), grid)
    out = out.view(B, T)  # [B x T]

    return out


def _create_params_matrix(sis: Tensor, pis: Tensor) -> Tensor:
    n = sis.size(0)
    theta = sis.new_zeros(torch.Size([n, 3]))

    s = sis.clone()
    x = pis.clone()
    # y = 0

    theta[:, 0] = s.view(-1)
    theta[:, 1] = x.view(-1)
    theta[:, 2] = 0
    return theta.float()


def _create_theta(params: Tensor) -> Tensor:
    # Takes 3-dimensional vectors, and massages them into 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    n = params.size(0)
    expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3]).to(params.device)
    out = torch.cat((params.new_zeros([1, 1]).expand(n, 1), params), 1)
    return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)


# noinspection PyPep8Naming
def _normalize_scale(T: int, sis: Tensor) -> Tensor:
    return T / sis


# noinspection PyPep8Naming
def _normalize_location(T: int, pis: Tensor, sis: Tensor) -> Tensor:
    """
    Normalizes the absolute value of z_where to the range that is appropriate for the network.
    :param T:
    :param pis:
    :param sis: unnormalized z_size
    :return:
    """
    x = pis.clone()
    x += sis / 2
    x -= T / 2
    x /= -(sis / 2)

    return x

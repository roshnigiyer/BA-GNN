import torch

def upsize_rows(x, rows, cols, add_row):
    first_half = x[0:int(rows/2), :]
    second_half = x[int(rows/2):, :]
    new_rows = torch.zeros(add_row, cols)
    new_x = torch.cat([first_half, new_rows, second_half], dim=0)
    return new_x


def upsize_cols(x, rows, cols, add_col):
    first_half = x[:, 0:int(cols/2)]
    second_half = x[:, int(cols/2):]
    new_cols = torch.zeros(rows, add_col)
    new_x = torch.cat([first_half, new_cols, second_half], dim=1)
    return new_x


def downsize_rows(x, rows, cols, remove_row):
    first_half = x[0:int(rows/7), :]
    second_half = x[int(rows/7)+remove_row:, :]
    new_x = torch.cat([first_half, second_half], dim=0)
    return new_x


def downsize_cols(x, rows, cols, remove_col):
    first_half = x[:, 0:int(cols/7)]
    second_half = x[:, int(cols/7)+remove_col:]
    new_x = torch.cat([first_half, second_half], dim=1)
    return new_x


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())
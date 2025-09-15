# %%
import pytorch_lightning
import torch

# %%


def createS():
    # These are the indices.  They need to be integers.
    # 2 rows and n columns
    i = torch.tensor([[0, 1, 1],
                      [2, 0, 2]])
    # n scalars in a vector with which to fill in the sparse matrix
    v = torch.tensor([3.0, 4.0, 5.0])
    # This needs i, v, and an explicit size, since you can't
    # infer the size (some row or column might be all zero)
    S = torch.sparse_coo_tensor(i, v, (2, 3),
                                requires_grad=True)
    return S


# %% This is a column vector we can apply s to
b = torch.tensor([[3.0], [4.0], [5.0]],
                 requires_grad=False)

y = torch.tensor([[2.0], [1.0]],
                 requires_grad=False)

# %%
# This is a sparse dot product you can take a gradient
# of.  See this page for details:
# https://pytorch.org/docs/stable/sparse.html
S = createS()
yHat = torch.sparse.mm(S, b)
yHat

# %%
# This is the simplest possible example of a gradient descent with all of the
# pieces pulled apart.
S = createS()
for i in range(10):
    if S.grad is not None:
        S.grad.zero_()

    yHat = torch.sparse.mm(S, b)
    loss = torch.norm(y-yHat)
    print(loss)
    loss.backward()

    with torch.no_grad():
        gamma = 0.1
        S -= gamma*S.grad
        # Note
        #    S = S - gamma*S.grad
        # does not work since S will then be a new copy and not in place!

# %%
# Now we be a bit fancier and use an optimizer.  This is identical to the above
S = createS()
optimizer = torch.optim.SGD([S], lr=0.1)
for i in range(10):
    optimizer.zero_grad()
    yHat = torch.sparse.mm(S, b)
    loss = torch.norm(y-yHat)
    print(loss)
    loss.backward()

    optimizer.step()

# %%
# Let's play around with momentum
S = createS()
# The velocity vector for the optimization
V = createS()*0
for i in range(10):
    if S.grad is not None:
        S.grad.zero_()

    yHat = torch.sparse.mm(S, b)
    loss = torch.norm(y-yHat)
    print(loss)
    loss.backward()

    with torch.no_grad():
        gamma = 0.1
        mu = 0.1
        # This momentum implementation is from
        #   https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        V = mu*V + S.grad
        S -= gamma*V
        # Note
        #    S = S - gamma*S.grad
        # does not work since S will then be a new copy and not in place!

# %%

# %%

# %%

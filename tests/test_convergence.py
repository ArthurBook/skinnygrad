from skinnygrad import optim, tensors


def test_sgd_training_loop():
    a = tensors.Tensor.random_normal(1, 5, mean=5, var=2, requires_grad=False)
    b = tensors.Tensor.random_normal(5, 1, mean=5, var=2, requires_grad=True)
    opt = optim.SGD((a, b), lr=0.001)

    initial_loss, loss_val = None, None
    for _ in range(1000):  # Run for 100 iterations
        with opt:
            loss = ((a @ b) - 10) ** 2
            loss.backprop()

        loss_val = loss.sum().realize()
        assert isinstance(loss_val, float)
        if initial_loss is None:
            initial_loss = loss_val

    assert initial_loss is not None
    assert loss_val is not None
    assert 0 <= loss_val < 0.01 < initial_loss  # Ensure loss decreases and is not zero

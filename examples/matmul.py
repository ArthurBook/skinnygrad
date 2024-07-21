import visualization

from skinnygrad import config, llops, tensors


def plot_matmul_fwd(
    params_list: list,
    mat_list: list,
) -> None:
    visualizer = visualization.GraphVisualizer(f"static/matmul-forward.png")
    with config.Configuration(visualizer):
        with visualizer._graph as main_g:
            main_g.graph_attr.update(rankdir="BT")
            with main_g.subgraph(
                name="cluster_Forward",
                label="Matmul forward pass",
                fontcolor="white",
                fontsize="60.0",
                labeljust="l",
                shape="box",
            ) as g:
                visualizer._subgraph = g
                params = tensors.Tensor(params_list)
                mat = tensors.Tensor(mat_list)
                a = params @ mat


def plot_matmul_backward(
    params_list: list,
    mat_list: list,
) -> None:
    visualizer = visualization.GraphVisualizer(f"static/matmul-backward.png")
    with config.Configuration(visualizer):
        with visualizer._graph as main_g:
            main_g.graph_attr.update(rankdir="TB")
            params = tensors.Tensor(params_list, requires_grad=True)
            mat = tensors.Tensor(mat_list)
            a = params @ mat
            with main_g.subgraph(
                name="cluster_Backward",
                label="Matmul backward pass",
                fontcolor="white",
                fontsize="60.0",
                labeljust="l",
                shape="box",
            ) as g:
                visualizer._subgraph = g
                a[0, 0].backprop()


if __name__ == "__main__":
    params_list = [[1, 2], [3, 4]]
    mat_list = [[2, 0], [1, 2]]
    # plot_matmul_fwd(params_list, mat_list)
    plot_matmul_backward(params_list, mat_list)

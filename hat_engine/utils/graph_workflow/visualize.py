# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501


import os
import re
import tempfile

from hat_engine.utils.apply_func import _as_list

try:
    import graphviz
except ImportError:
    graphviz = None

from hat_engine.utils.graph_workflow.symbol import Symbol, get_operator_name

__all__ = ["save_viz_graph", "visualize_graph"]


def save_viz_graph(graph, path):
    """
    Save visualized graph.

    Args:
        graph: Visualized graph
        path: Render path.
    """
    assert graphviz is not None, "Please install graphviz first"
    assert isinstance(
        graph, graphviz.Digraph
    ), f"Expected {graphviz.Digraph}, but get {str(type(graph))}"
    output_prefix, ext = os.path.splitext(path)
    graph.format = ext[1:]
    graph.render(output_prefix, cleanup=True)


def visualize_graph(
    graph,
    variable_node_color="red",
    optional_variable_node_color="orange",
    constant_variable_node_color="pink",
    op_node_color="blue",
    output_node_color="greenyellow",
    render_op_module_name=True,
    edge_color="black",
    save_path=None,
    view=False,
    node_name_reg=".*",
):
    """
    Visualize :py:class:`Symbol`.

    Args:
        graph: Symbolic graph
        variable_node_color: :py:class:`Variable` node color, by default "red"
        optional_variable_node_color: :py:class:`OptionalVariable` node
            color, by default "orange"
        constant_variable_node_color: :py:class:`Constant` node color, by
            default "pink"
        op_node_color: Operator node color, by default "blue"
        output_node_color: Output node color, by default "greenyellow"
        render_op_module_name: Render operator module name, by default True
        edge_color: Edge color, by default "black"
        save_path: Save path, by default None
        view: View the graph, by default False
        node_name_reg: Node name regular expression, by default ".*"
    """
    assert graphviz is not None, "Please install graphviz first"

    assert isinstance(
        graph, Symbol
    ), f"Expected type {Symbol}, but get {str(type(graph))}"  # noqa

    dot = graphviz.Digraph()

    def _is_input_node(node):
        return (
            node.is_placeholder
            or node.is_optional_placeholder
            or node.is_constant
        )  # noqa

    def _get_node_attr(node):
        render_args = {}
        if _is_input_node(node) and node in graph._outputs:
            render_args = {"style": "filled", "fillcolor": output_node_color}
        else:
            render_args = {}
        is_placeholder_or_constant = False
        if node.is_placeholder:
            render_args["color"] = variable_node_color
        elif node.is_optional_placeholder:
            render_args["color"] = optional_variable_node_color
        elif node.is_constant:
            render_args["color"] = constant_variable_node_color
        else:
            if node in graph._outputs:
                color = output_node_color
            else:
                color = op_node_color
            render_args["color"] = color
            _op_name = get_operator_name(node.op, render_op_module_name)
            render_args["label"] = f"{node.name}\nop={_op_name}"  # noqa
            is_placeholder_or_constant = False
        return {
            "render_args": render_args,
            "is_placeholder_or_constant": is_placeholder_or_constant,
        }

    if node_name_reg == ".*":

        def _render_graph(node):
            render_attrs = _get_node_attr(node)

            render_args = render_attrs["render_args"]
            is_placeholder_or_constant = render_attrs[
                "is_placeholder_or_constant"
            ]

            dot.node(name=node.name, **render_args)

            if not is_placeholder_or_constant:
                real_inputs = set()

                def _fn(x):
                    real_inputs.add(x.name)

                node._map_args(node.args, _fn)
                node._map_args(node.kwargs, _fn)

                for value in node._inputs.values():
                    if value.name in real_inputs:
                        kwargs = {}
                    else:
                        # dashed line for wait node
                        kwargs = {"style": "dashed"}

                    dot.edge(value.name, node.name, color=edge_color, **kwargs)

        graph.post_order_dfs_visit(_render_graph)

    else:
        nodes_need_render = []

        def fvisit(node):
            nonlocal node_name_reg
            for reg_ex in _as_list(node_name_reg):
                if re.match(reg_ex, node.name):
                    return nodes_need_render.append(node)

        graph.post_order_dfs_visit(fvisit)

        from hat_engine.utils.graph_workflow.graph_algo import (
            post_order_dfs_visit,
        )

        for current_node in nodes_need_render:
            tmp_nodes = []

            def fvisit(node):
                if node == current_node:
                    return
                if _is_input_node(node):
                    return tmp_nodes.append(node)
                if node in nodes_need_render:
                    return tmp_nodes.append(node)

            def findegree(node):
                if node in nodes_need_render and node != current_node:
                    return 0
                return node.num_inputs

            def finput(node, idx):
                key = list(node._inputs.keys())[idx]
                return node._inputs[key]

            post_order_dfs_visit(
                current_node,
                visit_fn=fvisit,
                indegree_fn=findegree,
                get_input_fn=finput,
            )

            for node in tmp_nodes:
                render_args = _get_node_attr(node)["render_args"]
                dot.node(name=node.name, **render_args)
                dot.edge(node.name, current_node.name, color=edge_color)

    if save_path is not None:
        save_viz_graph(dot, save_path)

    if view:
        if save_path is None:
            save_path = tempfile.NamedTemporaryFile(suffix=".png").name
            save_viz_graph(dot, save_path)
            cleanup = True
        else:
            cleanup = False

        import cv2
        from matplotlib import pyplot as plt

        plt.imshow(cv2.imread(save_path)[:, :, ::-1])

        if cleanup:
            os.remove(save_path)

    return dot

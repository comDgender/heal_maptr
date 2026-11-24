# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501


from collections import deque
from typing import Callable, Optional, Tuple

from hat_engine.utils.graph_workflow.symbol import Node, Symbol

__all__ = ["split_by_node_name", "split"]


def split_by_node_name(
    graph: Symbol,
    split_node_name: str,
) -> Tuple[Symbol]:
    """

    Split the graph into two parts by node name.

    Args:
        graph (Symbol): The graph to be split.
        split_node_name (str): The name of the node to be split.

    Returns:
        common_graph (Symbol): The graph before the split node.
        split_graph (Symbol): The graph after the split node.

    """

    def _is_split_node(node):
        return node.name == split_node_name

    return split(graph, _is_split_node)


def split(
    graph: Symbol,
    is_split_node: Callable[[Node], bool],
) -> Tuple[Symbol]:
    """

    Split the graph into two parts by node name or user defined funtion.

    Args:
        graph (Symbol): The graph to be split.
        is_split_node (Callable): Function to find a split node.

    Returns:
        common_graph (Symbol): The graph before the split node.
        split_graph (Symbol): The graph after the split node.

    """

    graph = graph.copy(deepcopy=False)
    created_place_holders = {}
    common_graph = set()

    def _split(node):
        """Fvisit function."""
        for input_i in node.inputs:
            if is_split_node(input_i):
                common_graph.add(input_i)
                if input_i.name not in created_place_holders:
                    created_place_holders[
                        input_i.name
                    ] = Node.create_placeholder(
                        name=input_i.name, attr=input_i.attr
                    )
                node._inputs[input_i] = created_place_holders[input_i.name]

    graph.post_order_dfs_visit(_split)

    return Symbol(list(common_graph)), graph


def get_split_node(
    graph: Symbol, start_node_name: Optional[str] = None
) -> Node:
    r"""Get the split node of the graph.

    split node: The node that has more than one output.

    Note:
        Due to the limitation of the current implementation,
        the split node encountered first will be returned.

        For a graph with multiple inputs, the first encoutered
        split node will not neccessarily the same if starting
        with different input node. Like the graph below,
        split_node_1 will be the first if start from input1;
        split_node_2 will be the first if start from input2.

          input1
            ||
        split_node_1   input2
         //    \\        ||
        n1      n2   split_node_2
                       //   \\
                      n3     n4

        Thus, for graph with more than
        one input nodes in graph, the argument start_node_name is
        required.

    Args:
        graph (Symbol): The graph to be split.
        start_node_name (str): The name of the node to start searching.

    """

    start_node = None
    if len(graph._roots) < 1:
        raise ValueError("graph has no input.")
    elif len(graph._roots) == 1:
        start_node = graph._roots[0]
    else:
        assert (
            start_node_name is not None
        ), "more than one input nodes in graph, start node name is required."
        for node in graph._roots:
            if node.name == start_node_name:
                start_node = node
                break
    assert start_node is not None, f"start node {start_node_name} not found."

    queue = deque([start_node])
    while queue:
        node = queue.popleft()
        if len(graph._leaves[node.name]) > 1 and node.is_callable:
            return node
        for output in graph._leaves[node.name]:
            queue.append(output)

import os
from typing import Dict

from et_replay.execution_trace import Node
import torch.distributed as dist


def init_hybrid_ep_buffer(node: Node, pg_groups: Dict[int, dist.ProcessGroup]):
    assert node.name == "HybridEPBuffer::__init__", "Node is not a HybridEPBuffer::__init__ node"

    try:
        import deep_ep
        import hybrid_ep_cpp
    except ImportError:
        raise ImportError("hubrid deep_ep is not installed")


    def to_hybrid_ep_dtype(dtype: str):
        if dtype == "uint16_t":
            return hybrid_ep_cpp.APP_TOKEN_DATA_TYPE.UINT16
        elif dtype == "uint8_t":
            return hybrid_ep_cpp.APP_TOKEN_DATA_TYPE.UINT8
        else:
            raise ValueError("Unsupported dtype: %s" % dtype)

    pg_id = node.inputs[0][0]

    config = hybrid_ep_cpp.BufferConfig()
    config.hidden_dim = node.inputs[0][1][0]
    config.max_num_of_tokens_per_rank = node.inputs[0][1][1]
    config.num_of_experts_per_rank = node.inputs[0][1][2]
    config.num_of_ranks_per_node = node.inputs[0][1][3]
    config.num_of_nodes = node.inputs[0][1][4]
    config.token_data_type = to_hybrid_ep_dtype(node.inputs[0][1][5])
    config.num_of_blocks_preprocessing_api = node.inputs[0][1][6]
    config.num_of_blocks_dispatch_api = node.inputs[0][1][7]
    config.num_of_blocks_combine_api = node.inputs[0][1][8]

    local_rank = node.inputs[0][2]
    node_rank = node.inputs[0][3]
    group_size = node.inputs[0][4]
    _ = node.inputs[0][5]
    load_cached_kernels = node.inputs[0][6]
    use_shared_buffer = node.inputs[0][7]
    enable_custom_allgather = node.inputs[0][8]

    return hybrid_ep_cpp.HybridEPBuffer(
        pg_groups[int(pg_id)],
        config, 
        local_rank, 
        node_rank, 
        group_size, 
        os.path.dirname(os.path.abspath(deep_ep.__file__)), 
        load_cached_kernels = load_cached_kernels,
        use_shared_buffer = use_shared_buffer,
        enable_custom_allgather = enable_custom_allgather
    )


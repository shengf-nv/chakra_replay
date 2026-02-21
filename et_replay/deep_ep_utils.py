
import os
from typing import List
from chakra_replay.et_replay.execution_trace import Node
import torch.distributed as dist

def init_hybrid_ep_buffer(node: Node, pg_groups: List[dist.ProcessGroup]):
    assert node.name == "HybridEPBuffer::__init__", "Node is not a HybridEPBuffer::__init__ node"

    try:
        import hybrid_ep_cpp
    except ImportError:
        raise ImportError("hubrid deep_ep is not installed")


    pg_id = node.inputs[0]


    config = hybrid_ep_cpp.BufferConfig()
    config.hidden_dim = node.inputs[1][0]
    config.max_num_of_tokens_per_rank = node.inputs[1][1]
    config.num_of_experts_per_rank = node.inputs[1][2]
    config.num_of_ranks_per_node = node.inputs[1][3]
    config.num_of_nodes = node.inputs[1][4]
    config.token_data_type = node.inputs[1][5]
    config.num_of_blocks_preprocessing_api = node.inputs[1][6]
    config.num_of_blocks_dispatch_api = node.inputs[1][7]
    config.num_of_blocks_combine_api = node.inputs[1][8]


    local_rank = node.inputs[2]
    node_rank = node.inputs[3]
    group_size = node.inputs[4]
    _ = node.inputs[5]
    load_cached_kernels = node.inputs[6]
    use_shared_buffer = node.inputs[7]
    enable_custom_allgather = node.inputs[8]

    my_buffer = hybrid_ep_cpp.HybridEPBuffer(
            group, 
            config, 
            local_rank, 
            node_rank, 
            group_size, 
            os.path.dirname(os.path.abspath(hybrid_ep_cpp.__file__)), 
            load_cached_kernels = load_cached_kernels,
            use_shared_buffer = use_shared_buffer,
            enable_custom_allgather = enable_custom_allgather
    )
    return my_buffer


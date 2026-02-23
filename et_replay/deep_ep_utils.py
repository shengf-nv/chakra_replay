import os
from typing import Dict

from et_replay.execution_trace import Node
import torch.distributed as dist

try:
    import deep_ep
    import hybrid_ep_cpp

    has_deep_ep = True
    has_hybrid_ep_cpp = True
except ImportError:
    has_deep_ep = False
    has_hybrid_ep_cpp = False

def to_hybrid_ep_dtype(dtype: str):
    if dtype == "uint16_t":
        return hybrid_ep_cpp.APP_TOKEN_DATA_TYPE.UINT16
    elif dtype == "uint8_t":
        return hybrid_ep_cpp.APP_TOKEN_DATA_TYPE.UINT8
    else:
        raise ValueError("Unsupported dtype: %s" % dtype)

def init_hybrid_ep_buffer(node: Node, pg_groups: Dict[int, dist.ProcessGroup]):
    assert node.name == "HybridEPBuffer::__init__", "Node is not a HybridEPBuffer::__init__ node"

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

def get_hybrid_ep_config_instance(node: Node):
    """Build HybridEpConfigInstance from trace node. Tuple order matches config.cuh to_ivalue_tuple() (23 elements)."""
    
    assert node.name.startswith("HybridEPBuffer::"), "Node is not a HybridEPBuffer:: node"
    inp = node.inputs[0][0]
    return hybrid_ep_cpp.HybridEpConfigInstance(
        hidden_dim=inp[0],
        max_num_of_tokens_per_rank=inp[1],
        num_of_experts_per_rank=inp[2],
        num_of_ranks_per_node=inp[3],
        num_of_nodes=inp[4],
        num_of_threads_per_block_preprocessing_api=inp[5],
        num_of_blocks_preprocessing_api=inp[6],
        num_of_blocks_permute_api=inp[7],
        token_data_type=to_hybrid_ep_dtype(inp[8]),
        num_of_stages_dispatch_api=inp[9],
        num_of_in_flight_s2g_dispatch_api=inp[10],
        num_of_tokens_per_chunk_dispatch_api=inp[11],
        num_of_blocks_dispatch_api=inp[12],
        forward_dispatch_api=bool(inp[13]),
        device_side_sync_dispatch_api=bool(inp[14]),
        num_of_stages_g2s_combine_api=inp[15],
        num_of_stages_s2g_combine_api=inp[16],
        num_of_tokens_per_chunk_combine_api=inp[17],
        num_of_tokens_per_group_combine_api=inp[18],
        num_of_blocks_combine_api=inp[19],
        num_of_additional_in_flight_s2g_combine_api=inp[20],
        backward_combine_api=bool(inp[21]),
        device_side_sync_combine_api=bool(inp[22]),
    )

def build_hybrid_ep_func(hybrid_ep_buffer: hybrid_ep_cpp.HybridEPBuffer, node: Node):
    assert node.name.startswith("HybridEPBuffer::"), "Node is not a HybridEPBuffer:: node"

    if node.name == "HybridEPBuffer::combine":
        def combine(hybrid_ep_buffer: hybrid_ep_cpp.HybridEPBuffer, args, kwargs):
            return hybrid_ep_buffer.combine(args, kwargs)
        return combine, 2
    elif node.name == "HybridEPBuffer::dispatch":
        return hybrid_ep_cpp.HybridEPBuffer.dispatch, 3
    elif node.name == "HybridEPBuffer::dispatch_with_permute":
        return hybrid_ep_cpp.HybridEPBuffer.dispatch_with_permute, 6
    elif node.name == "HybridEPBuffer::combine_with_unpermute":
        return hybrid_ep_cpp.HybridEPBuffer.combine_with_unpermute, 2
    else:
        raise ValueError("hybrid_ep_cpp: unsupported node name %s" % node.name) 
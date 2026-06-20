import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

def get_kvo_cache_info(unet: UNet2DConditionModel, height=512, width=512):
    latent_height = height // 8
    latent_width = width // 8
    
    kvo_cache_shapes = []
    kvo_cache_structure = []
    current_h, current_w = latent_height, latent_width
    
    for _, block in enumerate(unet.down_blocks):
        if hasattr(block, 'attentions') and block.attentions is not None:
            block_structure = []
            for attn_block in block.attentions:
                attn_count = 0
                for transformer in attn_block.transformer_blocks:
                    attn = transformer.attn1
                    hidden_dim = attn.to_k.out_features
                    seq_length = current_h * current_w
                    kvo_cache_shapes.append((seq_length, hidden_dim))
                    attn_count += 1
                block_structure.append(attn_count)
            kvo_cache_structure.append(block_structure)
        
        if hasattr(block, 'downsamplers') and block.downsamplers is not None:
            current_h //= 2
            current_w //= 2
    
    if hasattr(unet.mid_block, 'attentions') and unet.mid_block.attentions is not None:
        block_structure = []
        for attn_block in unet.mid_block.attentions:
            attn_count = 0
            for transformer in attn_block.transformer_blocks:
                attn = transformer.attn1
                hidden_dim = attn.to_k.out_features
                seq_length = current_h * current_w
                kvo_cache_shapes.append((seq_length, hidden_dim))
                attn_count += 1
            block_structure.append(attn_count)
        kvo_cache_structure.append(block_structure)
    
    for _, block in enumerate(unet.up_blocks):
        if hasattr(block, 'attentions') and block.attentions is not None:
            block_structure = []
            for attn_block in block.attentions:
                attn_count = 0
                for transformer in attn_block.transformer_blocks:
                    attn = transformer.attn1
                    hidden_dim = attn.to_k.out_features
                    seq_length = current_h * current_w
                    kvo_cache_shapes.append((seq_length, hidden_dim))
                    attn_count += 1
                block_structure.append(attn_count)
            kvo_cache_structure.append(block_structure)
        
        if hasattr(block, 'upsamplers') and block.upsamplers is not None:
            current_h *= 2
            current_w *= 2

    kvo_cache_count = sum(sum(block) for block in kvo_cache_structure)
    
    return kvo_cache_shapes, kvo_cache_structure, kvo_cache_count


def convert_list_to_structure(flat_list, structure):
    formatted_list = []
    flat_idx = 0
    for block_structure in structure:
        block_list = []
        for count in block_structure:
            layer_list = []
            for _ in range(count):
                if flat_idx >= len(flat_list):
                    break
                layer_list.append(flat_list[flat_idx])
                flat_idx += 1
            block_list.append(layer_list)
        formatted_list.append(block_list)
    return formatted_list


def convert_structure_to_list(structured_list):
    flat_list = []
    for block_list in structured_list:
        for layer_list in block_list:
            for item in layer_list:
                flat_list.append(item)
    return flat_list


def create_kvo_cache(unet: UNet2DConditionModel, batch_size, cache_maxframes, height=512, width=512, 
                     device='cuda', dtype=torch.float16):
    kvo_cache_shapes, kvo_cache_structure, _ = get_kvo_cache_info(unet, height, width)
    
    kvo_cache = []
    for seq_length, hidden_dim in kvo_cache_shapes:
        cache_tensor = torch.zeros(
            2, cache_maxframes, batch_size, seq_length, hidden_dim,
            dtype=dtype, device=device
        )
        kvo_cache.append(cache_tensor)
    
    return kvo_cache, kvo_cache_structure
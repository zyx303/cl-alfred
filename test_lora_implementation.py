#!/usr/bin/env python3
"""
Test script for LoRA implementation in seq2seq_im_mask_lora.py
"""

import torch
import torch.nn as nn
import sys
import os
os.environ['ALFRED_ROOT'] = '/home/yongxi/work/cl-alfred'
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from models.model.seq2seq_im_mask_lora import LoRALayer

def test_lora_layer():
    """Test basic LoRA layer functionality"""
    print("Testing LoRA Layer...")
    
    
    # Create a LoRA layer
    lora = LoRALayer(in_features=128, out_features=64, rank=4, scale=1.0)
    
    # Test forward pass
    x = torch.randn(10, 128)
    output = lora(x)
    
    assert output.shape == (10, 64), f"Expected shape (10, 64), got {output.shape}"
    print("✓ LoRA Layer forward pass test passed")
    
    # Test parameter count
    total_params = sum(p.numel() for p in lora.parameters())
    expected_params = 128 * 4 + 4 * 64  # lora_A + lora_B
    assert total_params == expected_params, f"Expected {expected_params} params, got {total_params}"
    print(f"✓ LoRA Layer parameter count: {total_params} (expected: {expected_params})")

def test_lora_linear():
    """Test LoRA linear layer functionality"""
    print("\nTesting LoRA Linear...")
    
    from models.model.seq2seq_im_mask_lora import LoRALinear
    
    # Create original linear layer
    original = nn.Linear(128, 64)
    
    # Wrap with LoRA
    lora_linear = LoRALinear(original, rank=4, scale=1.0)
    
    # Test forward pass
    x = torch.randn(10, 128)
    output = lora_linear(x)
    
    assert output.shape == (10, 64), f"Expected shape (10, 64), got {output.shape}"
    print("✓ LoRA Linear forward pass test passed")
    
    # Check that original parameters are frozen
    assert not any(p.requires_grad for p in original.parameters()), "Original parameters should be frozen"
    print("✓ Original parameters are correctly frozen")
    
    # Check that LoRA parameters are trainable
    lora_params = [p for p in lora_linear.lora.parameters() if p.requires_grad]
    assert len(lora_params) > 0, "LoRA parameters should be trainable"
    print(f"✓ LoRA parameters are trainable: {len(lora_params)} parameters")

def test_parameter_efficiency():
    """Test that LoRA significantly reduces trainable parameters"""
    print("\nTesting Parameter Efficiency...")
    
    from models.model.seq2seq_im_mask_lora import LoRALinear
    
    # Create a large linear layer
    original = nn.Linear(1024, 1024)
    lora_linear = LoRALinear(original, rank=16, scale=1.0)
    
    # Count parameters
    original_params = sum(p.numel() for p in original.parameters())
    lora_params = sum(p.numel() for p in lora_linear.lora.parameters())
    
    print(f"Original parameters: {original_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Reduction factor: {original_params / lora_params:.1f}x")
    
    # LoRA should be much smaller
    assert lora_params < original_params * 0.1, "LoRA should use <10% of original parameters"
    print("✓ LoRA achieves significant parameter reduction")

def create_dummy_args():
    """Create dummy arguments for testing"""
    class Args:
        def __init__(self):
            self.demb = 128
            self.dhid = 256
            self.dframe = 512
            self.pframe = 300
            self.attn_dropout = 0.1
            self.hstate_dropout = 0.1
            self.actor_dropout = 0.1
            self.input_dropout = 0.1
            self.vis_dropout = 0.1
            self.lang_dropout = 0.1
            self.dec_teacher_forcing = True
            self.pm_aux_loss_wt = 0.1
            self.subgoal_aux_loss_wt = 0.1
            self.action_loss_wt = 1.0
            self.mask_loss_wt = 1.0
            self.gpu = False
            self.zero_goal = False
            self.zero_instr = False
            self.panoramic = True
            self.orientation = False
            # LoRA specific
            self.lora_rank = 4
            self.lora_scale = 1.0
            self.lora_dropout = 0.0
    
    return Args()

def create_dummy_vocab():
    """Create dummy vocabulary for testing"""
    class DummyVocab:
        def __init__(self):
            self.word2index = lambda x: 0
            self.index2word = lambda x: ['test'] * len(x) if isinstance(x, list) else 'test'
    
    class Vocab:
        def __init__(self):
            self.action_low = DummyVocab()
    
    return Vocab()

def test_lora_attention_modules():
    """Test LoRA attention modules"""
    print("\nTesting LoRA Attention Modules...")
    
    from models.model.seq2seq_im_mask_lora import LoRASelfAttn, LoRAScaledDotAttn
    
    # Test LoRA Self Attention
    self_attn = LoRASelfAttn(dhid=256, lora_rank=4)
    x = torch.randn(2, 10, 256)  # batch_size=2, seq_len=10, hidden_dim=256
    output = self_attn(x)
    assert output.shape == (2, 256), f"Expected shape (2, 256), got {output.shape}"
    print("✓ LoRA Self Attention test passed")
    
    # Test LoRA Scaled Dot Attention
    scaled_attn = LoRAScaledDotAttn(dim_key_in=256, dim_key_out=64, 
                                   dim_query_in=256, dim_query_out=64, lora_rank=4)
    value = torch.randn(2, 10, 256)
    h = torch.randn(2, 256)
    output, scores = scaled_attn(value, h)
    assert output.shape == (2, 256), f"Expected output shape (2, 256), got {output.shape}"
    assert scores.shape == (2, 10), f"Expected scores shape (2, 10), got {scores.shape}"
    print("✓ LoRA Scaled Dot Attention test passed")

if __name__ == "__main__":
    print("Starting LoRA Implementation Tests...\n")
    
    try:
        test_lora_layer()
        test_lora_linear() 
        test_parameter_efficiency()
        test_lora_attention_modules()
        
        print("\n🎉 All tests passed! LoRA implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

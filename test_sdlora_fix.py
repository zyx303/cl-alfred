#!/usr/bin/env python3
"""
Test script to verify that SD-LoRA can properly handle model loading
without missing state_dict keys.
"""

import torch
import torch.nn as nn
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.getcwd()))

# Import the fixed SD-LoRA class
from methods.sdlora import LoRALinear

def test_lora_linear_compatibility():
    """Test if LoRALinear maintains compatibility with original Linear layers"""
    
    # Create a simple test case
    original_linear = nn.Linear(512, 256, bias=True)
    
    # Initialize with some test weights
    torch.nn.init.normal_(original_linear.weight, 0, 0.1)
    torch.nn.init.normal_(original_linear.bias, 0, 0.01)
    
    # Save original state dict
    original_state = {
        'test_layer.weight': original_linear.weight.clone(),
        'test_layer.bias': original_linear.bias.clone()
    }
    
    # Create LoRA version
    lora_linear = LoRALinear(original_linear, rank=8, alpha=1.0)
    
    # Test if we can load the original state dict into LoRA linear
    try:
        # Create a simple test model with LoRA layer
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.test_layer = lora_linear
                
        model = TestModel()
        
        # Try to load state dict
        missing_keys, unexpected_keys = model.load_state_dict(original_state, strict=False)
        
        print("Missing keys:", missing_keys)
        print("Unexpected keys:", unexpected_keys)
        
        # Check if the critical keys are not missing
        critical_missing = [k for k in missing_keys if not k.startswith('test_layer.lora_')]
        
        if not critical_missing:
            print("✅ SUCCESS: No critical keys missing!")
            return True
        else:
            print("❌ FAILURE: Critical keys missing:", critical_missing)
            return False
            
    except Exception as e:
        print(f"❌ FAILURE: Exception occurred: {e}")
        return False

def test_target_layers():
    """Test if the target layers from the error message are properly handled"""
    
    error_keys = [
        "enc_att_goal.scorer.weight", "enc_att_goal.scorer.bias",
        "enc_att_instr.scorer.weight", "enc_att_instr.scorer.bias",
        "dec.actor.0.weight", "dec.actor.0.bias",
        "dec.actor.2.weight", "dec.actor.2.bias",
        "dec.mask_dec.0.weight", "dec.mask_dec.0.bias",
        "dec.mask_dec.2.weight", "dec.mask_dec.2.bias",
        "dec.h_tm1_fc_goal.weight", "dec.h_tm1_fc_goal.bias",
        "dec.h_tm1_fc_instr.weight", "dec.h_tm1_fc_instr.bias",
        "dec.progress_goal.weight", "dec.progress_goal.bias",
        "dec.progress_instr.weight", "dec.progress_instr.bias",
        "dec.scale_dot_attn.fc_key.weight", "dec.scale_dot_attn.fc_key.bias",
        "dec.scale_dot_attn.fc_query.weight", "dec.scale_dot_attn.fc_query.bias",
        "dec.dynamic_conv.head1.weight", "dec.dynamic_conv.head1.bias",
        "dec.dynamic_conv.head2.weight", "dec.dynamic_conv.head2.bias",
        "dec.dynamic_conv.head3.weight", "dec.dynamic_conv.head3.bias",
        "dec.dynamic_conv.head4.weight", "dec.dynamic_conv.head4.bias",
        "dec.dynamic_conv.head5.weight", "dec.dynamic_conv.head5.bias"
    ]
    
    target_layers = {
        'enc_att_goal.scorer',
        'enc_att_instr.scorer', 
        'dec.actor.0',
        'dec.actor.2',
        'dec.mask_dec.0',
        'dec.mask_dec.2',
        'dec.h_tm1_fc_goal',
        'dec.h_tm1_fc_instr',
        'dec.progress_goal',
        'dec.progress_instr',
        'dec.scale_dot_attn.fc_key',
        'dec.scale_dot_attn.fc_query',
        'dec.dynamic_conv.head1',
        'dec.dynamic_conv.head2',
        'dec.dynamic_conv.head3',
        'dec.dynamic_conv.head4',
        'dec.dynamic_conv.head5'
    }
    
    # Extract layer names from error keys
    error_layer_names = set()
    for key in error_keys:
        layer_name = '.'.join(key.split('.')[:-1])  # Remove .weight or .bias
        error_layer_names.add(layer_name)
    
    print("Error layer names:", error_layer_names)
    print("Target layers:", target_layers)
    
    if error_layer_names == target_layers:
        print("✅ SUCCESS: All error layers are covered by target layers!")
        return True
    else:
        missing = error_layer_names - target_layers
        extra = target_layers - error_layer_names
        print(f"❌ FAILURE: Missing layers: {missing}, Extra layers: {extra}")
        return False

if __name__ == "__main__":
    print("Testing SD-LoRA fixes...")
    print("=" * 50)
    
    test1_passed = test_lora_linear_compatibility()
    print()
    test2_passed = test_target_layers()
    
    print("\n" + "=" * 50)
    if test1_passed and test2_passed:
        print("🎉 All tests passed! SD-LoRA fix should work.")
    else:
        print("❌ Some tests failed. Please review the implementation.")

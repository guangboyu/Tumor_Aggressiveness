#!/usr/bin/env python3
"""
Simple test script to verify the tumor classification pipeline works correctly.
"""

import os
import sys
import torch
import numpy as np
from dataset import TumorAggressivenessDataset
from model import TumorClassificationModel

def test_dataset():
    """Test the dataset loading."""
    print("Testing dataset loading...")
    
    # Test with smaller size for faster testing
    target_size = (32, 32, 32)
    
    try:
        # Test early fusion
        dataset = TumorAggressivenessDataset(
            csv_path='Data/ccRCC_Survival_Analysis_Dataset_english/training_set_603_cases.csv',
            ct_root='Data/data_nifty/1.Training_603',
            voi_root='Data/ROI/1.Training_ROI_603',
            ct_types=['A', 'D', 'N', 'V'],
            fusion_strategy='early',
            target_size=target_size,
            verbose=True
        )
        
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test first sample
        sample, label = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Label: {label}")
        print(f"Sample info: {dataset.get_sample_info(0)}")
        
        # Test label distribution
        label_dist = dataset.get_label_distribution()
        print(f"Label distribution: {label_dist}")
        
        # Analyze data completeness
        print("\nAnalyzing data completeness...")
        completeness = dataset.analyze_data_completeness()
        print(f"Total patients: {completeness['total_patients']}")
        print(f"Sequence availability: {completeness['sequence_availability']}")
        print(f"Patients with all sequences: {completeness['patients_with_all_sequences']}")
        print(f"Patients with no sequences: {completeness['patients_with_no_sequences']}")
        
        if completeness['patients_with_no_sequences'] > 0:
            print(f"Patients with no sequences: {list(completeness['missing_sequences_by_patient'].keys())[:5]}...")
        
        return True
        
    except Exception as e:
        print(f"Error testing dataset: {str(e)}")
        return False

def test_model():
    """Test the model creation and forward pass."""
    print("\nTesting model creation...")
    
    try:
        # Test early fusion model
        model = TumorClassificationModel(
            fusion_strategy='early',
            num_sequences=4,
            num_classes=2,
            model_depth=18
        )
        
        print(f"Model created successfully")
        print(f"Model info: {model.get_model_info()}")
        
        # Test forward pass
        batch_size = 2
        target_size = (32, 32, 32)
        
        # Test early fusion input
        early_input = torch.randn(batch_size, 4, *target_size)
        early_output = model(early_input)
        print(f"Early fusion input shape: {early_input.shape}")
        print(f"Early fusion output shape: {early_output.shape}")
        
        # Test intermediate fusion model
        intermediate_model = TumorClassificationModel(
            fusion_strategy='intermediate',
            num_sequences=4,
            num_classes=2,
            model_depth=18,
            fusion_method='attention'
        )
        
        intermediate_input = [torch.randn(batch_size, 1, *target_size) for _ in range(4)]
        intermediate_output = intermediate_model(intermediate_input)
        print(f"Intermediate fusion output shape: {intermediate_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        return False

def test_data_loader():
    """Test the data loader creation."""
    print("\nTesting data loader creation...")
    
    try:
        from dataset import TumorAggressivenessDataLoader
        
        loader = TumorAggressivenessDataLoader(
            data_root='Data',
            batch_size=2,
            target_size=(32, 32, 32),
            fusion_strategy='early'
        )
        
        train_loader = loader.create_train_loader()
        print(f"Train loader created successfully")
        print(f"Number of batches: {len(train_loader)}")
        
        # Test one batch
        for batch_idx, (data, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: data shape {data.shape}, labels {labels}")
            break
            
        return True
        
    except Exception as e:
        print(f"Error testing data loader: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("Starting pipeline tests...")
    
    # Test dataset
    dataset_ok = test_dataset()
    
    # Test model
    model_ok = test_model()
    
    # Test data loader
    loader_ok = test_data_loader()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY:")
    print(f"Dataset: {'✓ PASS' if dataset_ok else '✗ FAIL'}")
    print(f"Model: {'✓ PASS' if model_ok else '✗ FAIL'}")
    print(f"Data Loader: {'✓ PASS' if loader_ok else '✗ FAIL'}")
    
    if all([dataset_ok, model_ok, loader_ok]):
        print("\n🎉 All tests passed! The pipeline is ready to use.")
        print("\nTo start training, run:")
        print("python train.py --fusion_strategy early --batch_size 4 --epochs 10")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 
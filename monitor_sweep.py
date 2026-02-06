#!/usr/bin/env python3
"""
Monitor and analyze SLURM hyperparameter sweep results.

Usage:
    # Check job status
    python monitor_sweep.py --status
    
    # Cancel all jobs in array
    python monitor_sweep.py --cancel 12345678
    
    # Analyze results (aggregate metrics)
    python monitor_sweep.py --analyze logs/
"""

import subprocess
import argparse
import re
from pathlib import Path
from collections import defaultdict
import json


def get_job_status(array_id=None):
    """Get status of all SLURM jobs (or specific array)."""
    
    cmd = ["squeue", "-u", "$USER", "--format=%i,%j,%t,%P,%L"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        lines = result.stdout.strip().split('\n')
        
        print("JOBID,NAME,STATE,PARTITION,TIME_LEFT")
        for line in lines[1:]:
            if not line.strip():
                continue
            print(line)
    except Exception as e:
        print(f"Error getting job status: {e}")


def cancel_job_array(array_id):
    """Cancel all jobs in a SLURM job array."""
    
    if not array_id:
        print("ERROR: Must specify array ID (from squeue output)")
        return
    
    cmd = f"scancel {array_id}"
    print(f"Canceling array: {cmd}")
    
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        subprocess.run(cmd, shell=True)
        print("Jobs canceled")
    else:
        print("Canceled")


def analyze_results(log_dir):
    """Parse .out files and extract key metrics."""
    
    log_dir = Path(log_dir)
    results = defaultdict(dict)
    
    for log_file in sorted(log_dir.glob("sweep-*.out")):
        # Extract task ID from filename (sweep-123456_5.out)
        match = re.search(r'sweep-\d+_(\d+)\.out', log_file.name)
        if not match:
            continue
        
        task_id = int(match.group(1))
        
        # Parse log file
        with open(log_file) as f:
            content = f.read()
        
        # Extract metrics (customize regex based on your output)
        # Looking for patterns like "Loss: 0.5234" or "Valid: 85.2%"
        
        valid_match = re.search(r'Valid codewords: ([\d.]+)%', content)
        loss_match = re.search(r'Final loss: ([\d.]+)', content)
        
        results[task_id]['valid_pct'] = float(valid_match.group(1)) if valid_match else None
        results[task_id]['final_loss'] = float(loss_match.group(1)) if loss_match else None
        results[task_id]['completed'] = 'exit code: 0' in content.lower()
    
    # Print results sorted by validity
    print("\n=== RESULTS ===")
    print(f"{'Task':>4} {'Valid %':>8} {'Loss':>8} {'Status':>10}")
    print("-" * 35)
    
    for task_id in sorted(results.keys()):
        metrics = results[task_id]
        valid = metrics['valid_pct']
        loss = metrics['final_loss']
        status = "✓" if metrics['completed'] else "✗"
        
        valid_str = f"{valid:.1f}%" if valid is not None else "N/A"
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        
        print(f"{task_id:4d} {valid_str:>8} {loss_str:>8} {status:>10}")


def show_cpu_usage():
    """Show current CPU/GPU usage in partition."""
    
    cmd = "sinfo -p rtx4090 --format=%N,%C,%m"
    print("Current partition usage:")
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor SLURM sweep jobs")
    parser.add_argument("--status", action="store_true", help="Show job status")
    parser.add_argument("--cancel", type=int, metavar="ARRAY_ID", help="Cancel array jobs")
    parser.add_argument("--analyze", metavar="LOG_DIR", help="Analyze results from log directory")
    parser.add_argument("--usage", action="store_true", help="Show partition usage")
    
    args = parser.parse_args()
    
    if args.status:
        get_job_status()
    elif args.cancel:
        cancel_job_array(args.cancel)
    elif args.analyze:
        analyze_results(args.analyze)
    elif args.usage:
        show_cpu_usage()
    else:
        parser.print_help()

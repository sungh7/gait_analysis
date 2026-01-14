#!/bin/bash
# Progress monitoring script for GAVD full extraction

echo "==================================="
echo "GAVD Full Extraction Progress Monitor"
echo "==================================="
echo ""

# Check if process is running
if ps aux | grep -q "[g]avd_feature_extractor_full.py"; then
    echo "✓ Process is RUNNING"
    
    # Show last progress line
    echo ""
    echo "Latest progress:"
    grep "Progress:" /data/gait/gavd_full_extraction.log | tail -1
    
    echo ""
    echo "Last 10 lines of log:"
    tail -n 10 /data/gait/gavd_full_extraction.log
else
    echo "✗ Process is NOT RUNNING"
    echo ""
    echo "Checking if completed:"
    if grep -q "COMPLETED" /data/gait/gavd_full_extraction.log; then
        echo "✓ EXTRACTION COMPLETED!"
        echo ""
        grep -A 20 "SUMMARY" /data/gait/gavd_full_extraction.log
    else
        echo "✗ Process may have crashed"
        echo ""
        echo "Last 20 lines:"
        tail -n 20 /data/gait/gavd_full_extraction.log
    fi
fi

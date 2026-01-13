#!/bin/bash
# Monitor batch processing progress

OUTPUT_FILE="/tmp/claude/-data-gait-2d-project/tasks/b85a698.output"
RESULTS_DIR="/data/gait/2d_project/batch_results"

echo "========================================================================"
echo " BATCH PROCESSING PROGRESS MONITOR"
echo "========================================================================"
echo ""

# Check if process is running
if ps aux | grep -q "[b]atch_process_all_subjects.py"; then
    echo "✓ Process is running"
    echo ""

    # Show last 30 lines of output
    echo "Recent output:"
    echo "------------------------------------------------------------------------"
    tail -30 "$OUTPUT_FILE" 2>/dev/null | grep -v "^$" | tail -20
    echo "------------------------------------------------------------------------"
    echo ""

    # Count completed subjects
    if [ -d "$RESULTS_DIR" ]; then
        completed=$(ls "$RESULTS_DIR"/*_comparison.png 2>/dev/null | wc -l)
        echo "📊 Progress: $completed/26 subjects completed"

        if [ $completed -gt 0 ]; then
            echo ""
            echo "Completed subjects:"
            ls "$RESULTS_DIR"/*_comparison.png 2>/dev/null | sed 's/.*\(S1_[0-9]*\).*/  ✓ \1/'
        fi
    fi

else
    echo "⚠️  Process not running"

    # Check if completed
    if [ -f "$RESULTS_DIR/batch_results_summary.csv" ]; then
        echo "✓ Batch processing COMPLETE!"
        echo ""
        echo "Results available in: $RESULTS_DIR/"

        completed=$(ls "$RESULTS_DIR"/*_comparison.png 2>/dev/null | wc -l)
        echo "Total subjects processed: $completed"
    else
        echo "Checking output file for errors..."
        tail -50 "$OUTPUT_FILE" 2>/dev/null
    fi
fi

echo ""
echo "========================================================================"

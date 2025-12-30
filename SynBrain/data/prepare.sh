#!/bin/bash

# Run data preparation for remaining subjects: 2, 5, 7
for sub in 2 5 7; do
    echo "------------------------------------------------"
    echo "Starting data preparation for Subject $sub..."
    echo "------------------------------------------------"
    python prepare_nsddata_scale.py --sub $sub --session 40
    
    if [ $? -ne 0 ]; then
        echo "Error processing Subject $sub"
        exit 1
    fi
    echo "Finished Subject $sub"
done

echo "All subjects processed successfully."

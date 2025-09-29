#!/bin/bash

# Create 10 destination directories
for i in {1..10}; do
    mkdir -p "dir$i"
done

# Count files and calculate files per directory
total_files=$(ls -1 | grep -v "dir" | wc -l)
files_per_dir=$((total_files / 10))

# Move files to directories
count=0
dir_num=1

for file in $(ls -1 | grep -v "dir"); do
    # Move file to current directory
    mv "$file" "dir$dir_num/"
    
    # Increment counter
    count=$((count + 1))
    
    # Check if we need to move to next directory
    if [ $count -eq $files_per_dir ] && [ $dir_num -lt 10 ]; then
        dir_num=$((dir_num + 1))
        count=0
    fi
done


#!/bin/bash

# This script packages all release runs into a tarball with a git SHA so that we can upload to our
# release page. The SHA is to avoid conflicts with previous runs, but when we upload we should
# remove that so that users can expect that the name is release_runs.tar.gz (this renaming can be
# done in the Github Release UI).

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/..)
cd $PROJECT_ROOT

set -eou pipefail

# Create a temporary directory
TMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TMP_DIR"

# Loop over all the recipe runs and package them into a tarball
for tbevent in $(ls code_snapshots/*/recipes/**/logs/*/tensorboard/events*); do
    exp_name=$(basename -- $(cut -d/ -f3 <<<$tbevent) -logs)
    # Obfuscate the hostname
    # events.out.tfevents.1744822578.<host-name>.780899.0
    obfuscated_event_path=$(basename $tbevent | awk -F. '{print $1"."$2"."$3"."$4".HOSTNAME."$(NF-1)"."$NF}')
    
    # Create subdirectory for experiment if it doesn't exist
    mkdir -p "$TMP_DIR/$exp_name"
    
    # Copy the event file with obfuscated name to the experiment subdirectory
    cp "$tbevent" "$TMP_DIR/$exp_name/$obfuscated_event_path"
    
    echo "[$exp_name] Copied $tbevent to $TMP_DIR/$exp_name/$obfuscated_event_path"
done

# Create a tarball of all the processed event files
OUTPUT_TAR="release_runs-$(git rev-parse --short HEAD).tar.gz"
tar -czf "$OUTPUT_TAR" -C "$TMP_DIR" .
echo "Created tarball: $OUTPUT_TAR"

# Clean up the temporary directory
rm -rf "$TMP_DIR"
echo "Cleaned up temporary directory $TMP_DIR"

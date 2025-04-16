# Tests

## Launching Release Tests

```sh
# Assuming in NeMo RL project root

cd tools/

IS_RELEASE=1 CONTAINER=... ACCOUNT=... PARTITION=... ./launch <script_path> <another_script_path> ...

# DRYRUN=1 to get a rough estimate of compute
DRYRUN=1 IS_RELEASE=1 CONTAINER=... ACCOUNT=... PARTITION=... ./launch <script_path> <another_script_path> ...

# DRYRUN=2 will create a codesnapshot with a fully hermetic example
DRYRUN=2 IS_RELEASE=1 CONTAINER=... ACCOUNT=... PARTITION=... ./launch <script_path> <another_script_path> ...

# Run all (Caution: this will use a lot of compute; consider listing out the jobs)
IS_RELEASE=1 CONTAINER=... ACCOUNT=... PARTITION=... ./launch ../../recipes/**/*.sh
```

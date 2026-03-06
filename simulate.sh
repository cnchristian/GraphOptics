#!/bin/bash

# TODO need to finish automatic creation of micromamba environment with
# TODO Need to flesh out exactly how stuff that needs internet will work
    # TODO can we provide an easy way to predownload stuff that does not require user input?

export PATH=$PATH:/mnt/c/Windows/System32/WindowsPowerShell/v1.0/

# ===============================================================
# Remote Slurm Submission Script with Adaptive Per-Job Watcher
# ===============================================================

# --- User Configuration ---
REMOTE_USER="christia"
REMOTE_HOST="kuma.hpc.epfl.ch"
LOCAL_SRC="src/"
PROJECT_ROOT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(basename "$PROJECT_ROOT_PATH")
LOCAL_RESULTS_DIR="$PROJECT_ROOT_PATH/results"
JOBS_DIR="$PROJECT_ROOT_PATH/jobs"
WATCH_LOG="$PROJECT_ROOT_PATH/watch_$JOB_ID.log"

mkdir -p "$LOCAL_RESULTS_DIR" "$JOBS_DIR"

# ===============================================================
# Adaptive Watcher Function
# ===============================================================
adaptive_watcher() {
    local JOBINFO_FILE="$1"
    local JOBINFO_PATH="$PROJECT_ROOT_PATH/$JOBINFO_FILE"
    source "$JOBINFO_PATH" || { echo "[Watcher] Missing $JOBINFO_FILE"; exit 1; }

    local START_TIME_FILE="$JOBINFO_PATH.start"
    if [ ! -f "$START_TIME_FILE" ]; then
        date +%s > "$START_TIME_FILE"
    fi
    local START_TIME=$(cat "$START_TIME_FILE")
    local NOW=$(date +%s)
    local ELAPSED=$((NOW - START_TIME))

    echo "[Watcher] Checking job $JOB_ID (elapsed: ${ELAPSED}s)..."

    local STATUS=$(ssh "$REMOTE_USER@$REMOTE_HOST" "sacct -j $JOB_ID --format=State --noheader | head -n1 | awk '{print \$1}'")

    if [[ "$STATUS" == "COMPLETED" ]]; then
        echo "[Watcher] ✅ Job $JOB_ID completed. Downloading results..."
        mkdir -p "$LOCAL_RESULTS_DIR/out-$TIMESTAMP"
        rsync -avz "$REMOTE_USER@$REMOTE_HOST:$REMOTE_OUT/" "$LOCAL_RESULTS_DIR/out-$TIMESTAMP/"

        echo "[Watcher] Cleaning up remote files..."
        ssh "$REMOTE_USER@$REMOTE_HOST" "rm -rf '$REMOTE_OUT' '$REMOTE_SRC'"

        echo "[Watcher] Removing job info..."
        rm -f "$JOBINFO_PATH" "$START_TIME_FILE"

        # Create a Windows toast notification via BurntToast
        powershell.exe -Command "Import-Module BurntToast; \$winPath = '$(wslpath -w "$LOCAL_RESULTS_DIR/out-$TIMESTAMP")'; \$urlPath = \$winPath -replace '\\\\','/' -replace ' ','%20'; New-BurntToastNotification -Text 'Job Completed','Job $JOB_ID has finished' -Sound Default -Button (New-BTButton -Content 'View Results' -Arguments \$urlPath)"

        # Remove cron entry for this watcher
        crontab -l | grep -v "$JOBINFO_FILE" | crontab -
        echo "[Watcher] ✅ Done and cron entry removed."
        exit 0

    elif [[ "$STATUS" == "FAILED" || "$STATUS" == "CANCELLED" || "$STATUS" == "CANCELLED+" ]]; then
        echo "[Watcher] ❌ Job $JOB_ID failed ($STATUS)"

        mkdir -p "$LOCAL_RESULTS_DIR/out-$TIMESTAMP"
        rsync -avz "$REMOTE_USER@$REMOTE_HOST:$REMOTE_OUT/" "$LOCAL_RESULTS_DIR/out-$TIMESTAMP/"

        echo "[Watcher] Cleaning up remote files..."
        ssh "$REMOTE_USER@$REMOTE_HOST" "rm -rf '$REMOTE_OUT' '$REMOTE_SRC'"

        echo "[Watcher] Removing job info..."
        rm -f "$JOBINFO_PATH" "$START_TIME_FILE"

        powershell.exe -Command "Import-Module BurntToast; \$winPath = '$(wslpath -w "$LOCAL_RESULTS_DIR/out-$TIMESTAMP")'; \$urlPath = \$winPath -replace '\\\\','/' -replace ' ','%20'; New-BurntToastNotification -Text 'Job Failed','Job $JOB_ID has failed' -Sound Default -Button (New-BTButton -Content 'View Results' -Arguments \$urlPath)"
        rm -f "$JOBINFO_PATH" "$START_TIME_FILE"
        crontab -l | grep -v "$JOBINFO_FILE" | crontab -
        exit 0

    else
        echo "[Watcher] Job $JOB_ID still $STATUS."
        # Determine next polling interval (seconds)
        local NEXT_DELAY
        if (( ELAPSED < 3600 )); then           # < 1 hour
            NEXT_DELAY=1                        # every 1 minute
        elif (( ELAPSED < 10800 )); then        # < 3 hours
            NEXT_DELAY=5                        # every 5 minutes
        elif (( ELAPSED < 21600 )); then        # < 6 hours
            NEXT_DELAY=10                       # every 10 minutes
        else                                    # > 6 hours
            NEXT_DELAY=20                       # every 20 minutes
        fi
        echo "[Watcher] Rescheduling to check again in $NEXT_DELAY min."

        # Re-add this watcher to cron with new delay
        TMPCRON=$(mktemp)
        crontab -l > "$TMPCRON" 2>/dev/null || true
        # Remove old instance if present
        grep -v "$JOBINFO_FILE" "$TMPCRON" > "${TMPCRON}_new"
        mv "${TMPCRON}_new" "$TMPCRON"
        echo "*/$NEXT_DELAY * * * * bash $(realpath "$0") --watch '$JOBINFO_FILE' >> $PROJECT_ROOT_PATH/watch_${JOB_ID}.log 2>&1" >> "$TMPCRON"
        crontab "$TMPCRON"
        rm "$TMPCRON"
        echo "[Watcher] ✅ Rescheduled."
        exit 0
    fi
}

# ===============================================================
# Command Routing
# ===============================================================
if [[ "$1" == "--watch" ]]; then
    shift
    adaptive_watcher "$1"
    exit 0
fi

# ===============================================================
# Job Submission
# ===============================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REMOTE="/scratch/$REMOTE_USER/$PROJECT_ROOT"
REMOTE_SRC="$REMOTE/${LOCAL_SRC%/}-$TIMESTAMP/"                    # Remote directory
REMOTE_OUT="$REMOTE/out-$TIMESTAMP/"
REMOTE_DATA="$REMOTE/data"
REMOTE_SCRIPT="${REMOTE_SRC%/}/main.py"  # Script to run remotely

# --- Step 1: Copy local directory to remote ---
echo "Creating directories on $REMOTE_HOST..."
ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p '$REMOTE_SRC' '$REMOTE_OUT'"
echo "Copying $LOCAL_SRC to $REMOTE_USER@$REMOTE_HOST:$REMOTE_SRC"
rsync -avz "$LOCAL_SRC" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_SRC"

# --- Step 2: SSH into remote and run script, forwarding all arguments ---
args=$(printf "'%s' " "$@")
echo "Executing $REMOTE_SCRIPT on $REMOTE_HOST with args: $args"
JOB_SUBMISSION=$(ssh "$REMOTE_USER@$REMOTE_HOST" bash -s <<REMOTE_CMDS
#!/bin/bash
echo "This is the beginning of the job"

# === Environment setup ===
#set -e
echo "Set exit mode"

source ~/.bashrc
echo $?
echo "Set source"

module load gcc/13.2.0 cuda/12.4.1 || true  # Load CUDA if available
echo $?
echo "Loaded module successfully"


# === Step 0: Make sure data folder exists ===
if [ -d "$REMOTE_DATA" ]; then
    echo "[INFO] '$REMOTE_DATA' already exists. Assuming datasets are ready."
else
    echo "[INFO] '$REMOTE_DATA' not found. Creating it..."
    mkdir -p "$REMOTE_DATA"

    echo "[INFO] Calling Python to download datasets..."
    source ~/.bashrc
    micromamba activate OGS
    python3 - <<EOF
from torchvision import datasets

data_root = "${REMOTE_DATA}"

dataset_list = [
    datasets.MNIST,
    datasets.FashionMNIST,
    datasets.Imagenette,
    datasets.CIFAR10
]

print("[INFO] Starting dataset downloads...\n")

for ds in dataset_list:
    try:
        print(f"[INFO] Downloading {ds.__name__}...")
        ds(root=data_root, download=True)
        print(f"[OK] {ds.__name__} downloaded.")
    except Exception as e:
        print(f"[ERROR] Failed downloading {ds.__name__}: {e}")

print("\n[INFO] Dataset download phase complete.")
EOF
    micromamba deactivate
fi

## === Step 1: Detect Python dependencies ===
#echo "[INFO] Scanning project for Python imports..."
#
#python - <<PYCODE "$REMOTE" > "$REMOTE/.imports.txt"
#import ast, os, sys
#project_dir = sys.argv[1]
#imports = set()
#for root, _, files in os.walk(project_dir):
#    for f in files:
#        if f.endswith(".py" ):
#            path = os.path.join(root, f)
#            try:
#                with open(path, encoding="utf-8") as src:
#                    tree = ast.parse(src.read(), filename=path)
#                for node in ast.walk(tree):
#                    if isinstance(node, ast.Import):
#                        for n in node.names:
#                            imports.add(n.name.split('.')[0])
#                    elif isinstance(node, ast.ImportFrom) and node.module:
#                        imports.add(node.module.split('.')[0])
#            except Exception as e:
#                print(f"[WARN] Failed to parse {path}: {e}", file=sys.stderr)
#stdlibs = {
#    "sys","os","re","math","json","time","argparse","logging","typing",
#    "subprocess","pathlib","collections","itertools","functools","datetime",
#    "random","traceback","dataclasses","tempfile","shutil","glob","copy","enum",
#    "inspect","pprint","contextlib","threading","multiprocessing","asyncio",
#    "warnings","typing_extensions","importlib","io"
#}
#filtered = sorted(pkg for pkg in imports if pkg not in stdlibs and not pkg.startswith('_'))
#print("\n".join(filtered))
#PYCODE
#
#echo "[INFO] Detected imports:"
#cat "$REMOTE/.imports.txt" || echo "(none)"
#
## === Step 2: Build environment.yml ===
#echo "[INFO] Building environment.yml..."
#
#{
#echo "name: $PROJECT_ROOT"
#echo "channels:"
#echo "  - nvidia"
#echo "  - pytorch"
#echo "  - conda-forge"
#echo "dependencies:"
#echo "  - python==3.11"
#echo "  - pip"
#echo "  - numpy"
#echo "  - matplotlib"
#echo "  - scikit-learn"
#echo "  - seaborn"
#echo "  - pip:"
#echo "    - torch"
#echo "    - torchvision"
#
#} > "$REMOTE/environment.yml"
#
## === Step 3: Create or update micromamba environment ===
#if micromamba env list | grep -q "$PROJECT_ROOT"; then
#    echo "[INFO] Environment already exists"
#else
#    echo "[INFO] Creating new environment..."
#    micromamba env create -n "$PROJECT_ROOT" -f "$REMOTE/environment.yml" --yes
#    echo "[INFO] Created a new environment successfully"
#fi
#
## TODO there is an error in the env creation cause below this is never reached

# === Step 4: Submit Slurm job ===
echo "[INFO] Submitting Slurm job..."
sbatch <<SLURM_SCRIPT
#!/bin/bash
#SBATCH --partition h100
#SBATCH --qos normal
#SBATCH --cpus-per-task 8
#SBATCH --mem 32G
#SBATCH --gpus 1
#SBATCH --time 2-23:59:59
#SBATCH -o $REMOTE_OUT/job_%j.out
#SBATCH -e $REMOTE_OUT/job_%j.err

source ~/.bashrc
micromamba activate OGS
module load gcc/13.2.0 cuda/12.4.1

echo "[INFO] Running: $REMOTE_SCRIPT $args"
cd $REMOTE_OUT
python $REMOTE_SCRIPT $args
SLURM_SCRIPT
REMOTE_CMDS
)

echo "$JOB_SUBMISSION"

JOB_ID=$(echo "$JOB_SUBMISSION" | awk '/Submitted batch job/{print $4}')
echo "Job submitted: $JOB_ID"

JOB_INFO="jobs/job_${JOB_ID}_${TIMESTAMP}.jobinfo"
mkdir -p jobs
cat > "$JOB_INFO" <<EOF
JOB_ID=$JOB_ID
TIMESTAMP=$TIMESTAMP
REMOTE_OUT=$REMOTE_OUT
REMOTE_SRC=$REMOTE_SRC
REMOTE_HOST=$REMOTE_HOST
REMOTE_USER=$REMOTE_USER
EOF

# ===============================================================
# Schedule Initial Watcher
# ===============================================================
echo "Scheduling dedicated watcher for job $JOB_ID (every 1 min initially)..."
TMPCRON=$(mktemp)
crontab -l > "$TMPCRON" 2>/dev/null || true
grep -v "$JOB_INFO" "$TMPCRON" > "${TMPCRON}_new"
mv "${TMPCRON}_new" "$TMPCRON"
echo "*/1 * * * * bash $(realpath "$0") --watch '$JOB_INFO' >> $PROJECT_ROOT_PATH/watch_${JOB_ID}.log 2>&1" >> "$TMPCRON"
crontab "$TMPCRON"
rm "$TMPCRON"

echo "✅ Job $JOB_ID submitted and adaptive watcher registered."
echo "Results will appear in: $LOCAL_RESULTS_DIR/out_$TIMESTAMP/"

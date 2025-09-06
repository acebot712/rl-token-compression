#!/bin/bash
#
# One-command GCloud T4 instance launcher for RL Token Compression
# Usage: ./launch-gcloud.sh [command]
# Commands: create, ssh, stop, start, delete, status, run-quick, run-full

set -e

# Configuration
INSTANCE_NAME="rl-token-compression"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
ACCELERATOR="type=nvidia-tesla-t4,count=1"
BOOT_DISK_SIZE="100GB"
REPO_URL="${REPO_URL:-https://github.com/yourusername/rl-token-compression.git}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if gcloud is installed
check_gcloud() {
    # Try to find gcloud in common locations
    if command -v gcloud &> /dev/null; then
        GCLOUD_CMD="gcloud"
    elif [ -f "$HOME/google-cloud-sdk/bin/gcloud" ]; then
        GCLOUD_CMD="$HOME/google-cloud-sdk/bin/gcloud"
    else
        print_error "gcloud CLI not found. Install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
}

# Create startup script that will run on instance boot
create_startup_script() {
    cat > /tmp/startup-script.sh << 'EOF'
#!/bin/bash
# This runs automatically when instance starts

# Install dependencies
apt-get update
apt-get install -y python3 python3-pip python3-venv git build-essential wget

# Install NVIDIA drivers and CUDA
if ! nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    apt-get update
    apt-get install -y nvidia-driver-535 nvidia-utils-535
fi

# Create marker file when setup is complete
touch /tmp/setup-complete
EOF
}

# Create and setup instance with a single command
create_instance() {
    print_info "Creating T4 instance '$INSTANCE_NAME' in zone $ZONE..."
    
    # Check if instance already exists
    if $GCLOUD_CMD compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
        print_warning "Instance already exists. Connecting to it..."
        ssh_instance
        return 0
    fi
    
    # Create startup script
    create_startup_script
    
    # Create instance with startup script
    print_info "Creating instance with GPU..."
    $GCLOUD_CMD compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator=$ACCELERATOR \
        --image-family=debian-11 \
        --image-project=debian-cloud \
        --maintenance-policy=TERMINATE \
        --boot-disk-size=$BOOT_DISK_SIZE \
        --boot-disk-type=pd-balanced \
        --metadata-from-file startup-script=/tmp/startup-script.sh \
        --metadata="install-nvidia-driver=True" \
        --scopes=https://www.googleapis.com/auth/cloud-platform
    
    print_success "Instance created!"
    print_info "Waiting for instance to initialize (60 seconds)..."
    sleep 60
    
    # Setup repository and run training
    print_info "Setting up repository and starting training..."
    $GCLOUD_CMD compute ssh $INSTANCE_NAME --zone=$ZONE --command="
        # Clone repository
        if [ ! -d 'rl-token-compression' ]; then
            git clone $REPO_URL
        fi
        
        cd rl-token-compression
        
        # The make command handles everything else
        echo '====================================='
        echo 'Starting automatic training pipeline'
        echo '====================================='
        make pipeline-debug
    "
    
    print_success "Training completed! Check the results above."
}

# Simple one-command execution
one_command_run() {
    print_info "Starting one-command execution..."
    
    # Check if instance exists
    if $GCLOUD_CMD compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
        print_info "Instance exists. Starting training..."
        $GCLOUD_CMD compute ssh $INSTANCE_NAME --zone=$ZONE --command="
            cd rl-token-compression && make pipeline-debug
        "
    else
        # Create instance and run
        create_instance
    fi
}

# SSH into instance
ssh_instance() {
    print_info "Connecting to instance '$INSTANCE_NAME'..."
    $GCLOUD_CMD compute ssh $INSTANCE_NAME --zone=$ZONE
}

# Stop instance (save money)
stop_instance() {
    print_info "Stopping instance '$INSTANCE_NAME'..."
    $GCLOUD_CMD compute instances stop $INSTANCE_NAME --zone=$ZONE
    print_success "Instance stopped. Billing paused."
}

# Start instance
start_instance() {
    print_info "Starting instance '$INSTANCE_NAME'..."
    $GCLOUD_CMD compute instances start $INSTANCE_NAME --zone=$ZONE
    print_success "Instance started."
}

# Delete instance
delete_instance() {
    print_warning "This will permanently delete the instance!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $GCLOUD_CMD compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
        print_success "Instance deleted."
    fi
}

# Check status
status() {
    if $GCLOUD_CMD compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
        STATUS=$($GCLOUD_CMD compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(status)")
        print_info "Instance status: $STATUS"
        
        if [ "$STATUS" = "RUNNING" ]; then
            # Check GPU
            $GCLOUD_CMD compute ssh $INSTANCE_NAME --zone=$ZONE --command="nvidia-smi -L 2>/dev/null" &>/dev/null && \
                print_success "GPU is available" || \
                print_warning "GPU not detected"
        fi
    else
        print_info "No instance found."
    fi
}

# Main command handler
main() {
    check_gcloud
    
    # Default to one-command execution if no args
    if [ $# -eq 0 ]; then
        one_command_run
        exit 0
    fi
    
    case "$1" in
        create)
            create_instance
            ;;
        ssh)
            ssh_instance
            ;;
        stop)
            stop_instance
            ;;
        start)
            start_instance
            ;;
        delete)
            delete_instance
            ;;
        status)
            status
            ;;
        run|quick)
            one_command_run
            ;;
        help)
            echo "Usage: $0 [command]"
            echo ""
            echo "No arguments     - Create instance and run training automatically"
            echo "create           - Create instance and run training"
            echo "ssh              - Connect to instance"
            echo "stop             - Stop instance (save money)"
            echo "start            - Start instance"
            echo "delete           - Delete instance"
            echo "status           - Check status"
            echo ""
            echo "Simplest usage:"
            echo "  $0              # Does everything automatically"
            ;;
        *)
            one_command_run
            ;;
    esac
}

# Run main
main "$@"
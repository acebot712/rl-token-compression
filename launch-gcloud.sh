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
BOOT_DISK_SIZE="50GB"
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
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI not found. Install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
}

# Check credits (approximate)
check_credits() {
    print_info "Checking GCloud billing..."
    BILLING_ACCOUNT=$(gcloud billing accounts list --format="value(name)" --limit=1)
    if [ -z "$BILLING_ACCOUNT" ]; then
        print_warning "No billing account found. Make sure billing is enabled."
    else
        print_info "Active billing account: $BILLING_ACCOUNT"
        # Note: Actual credit balance requires billing API permissions
        print_info "To see detailed credits, visit: https://console.cloud.google.com/billing"
    fi
}

# Create T4 instance
create_instance() {
    print_info "Creating T4 instance '$INSTANCE_NAME' in zone $ZONE..."
    
    # Check if instance already exists
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
        print_warning "Instance already exists. Use 'start' or 'ssh' command."
        return 1
    fi
    
    gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --accelerator=$ACCELERATOR \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --boot-disk-size=$BOOT_DISK_SIZE \
        --boot-disk-type=pd-standard \
        --metadata="install-nvidia-driver=True" \
        --scopes=https://www.googleapis.com/auth/cloud-platform
    
    print_success "Instance created successfully!"
    print_info "Waiting for instance to be ready..."
    sleep 30
    
    # Install repo and dependencies automatically
    print_info "Setting up RL Token Compression on the instance..."
    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
        git clone $REPO_URL && \
        cd rl-token-compression && \
        echo '✓ Repository cloned' && \
        python3 -c 'import torch; print(f\"✓ GPU Available: {torch.cuda.is_available()}\")' && \
        echo '✓ Instance ready for training!'
    " || print_warning "Initial setup will complete on first SSH"
    
    print_success "Instance is ready! Use './launch-gcloud.sh ssh' to connect"
}

# SSH into instance
ssh_instance() {
    print_info "Connecting to instance '$INSTANCE_NAME'..."
    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE
}

# Stop instance (save money when not using)
stop_instance() {
    print_info "Stopping instance '$INSTANCE_NAME'..."
    gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE
    print_success "Instance stopped. Billing paused. Use 'start' to resume."
}

# Start instance
start_instance() {
    print_info "Starting instance '$INSTANCE_NAME'..."
    gcloud compute instances start $INSTANCE_NAME --zone=$ZONE
    print_success "Instance started. Use 'ssh' to connect."
}

# Delete instance completely
delete_instance() {
    print_warning "This will permanently delete the instance and all data!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Deleting instance '$INSTANCE_NAME'..."
        gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE --quiet
        print_success "Instance deleted."
    else
        print_info "Deletion cancelled."
    fi
}

# Check instance status
check_status() {
    print_info "Checking instance status..."
    if gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE &>/dev/null; then
        STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(status)")
        IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
        
        if [ "$STATUS" = "RUNNING" ]; then
            print_success "Instance is $STATUS (IP: $IP)"
            
            # Check if GPU is working
            gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="nvidia-smi -L" &>/dev/null && \
                print_success "GPU is available and working" || \
                print_warning "GPU status unknown"
        else
            print_info "Instance status: $STATUS"
        fi
    else
        print_info "No instance found. Use 'create' to create one."
    fi
}

# Run quick training pipeline
run_quick() {
    print_info "Running quick training pipeline (10 minutes)..."
    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
        cd rl-token-compression && \
        make pipeline-debug 2>&1 | tee training_quick.log && \
        echo '✓ Quick training completed! Check training_quick.log for results'
    "
}

# Run full training pipeline
run_full() {
    print_info "Running full training pipeline (this will take hours)..."
    print_warning "Consider using tmux/screen to keep training running if connection drops"
    
    gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="
        cd rl-token-compression && \
        nohup make pipeline-full > training_full.log 2>&1 & \
        echo 'Training started in background. Check progress with:' && \
        echo '  tail -f rl-token-compression/training_full.log'
    "
    
    print_success "Training started in background!"
    print_info "To check progress: ./launch-gcloud.sh ssh"
    print_info "Then run: tail -f rl-token-compression/training_full.log"
}

# Main command handler
main() {
    check_gcloud
    
    case "${1:-help}" in
        create)
            check_credits
            create_instance
            ;;
        ssh|connect)
            ssh_instance
            ;;
        stop)
            stop_instance
            ;;
        start)
            start_instance
            ;;
        delete|destroy)
            delete_instance
            ;;
        status)
            check_status
            ;;
        run-quick|quick)
            run_quick
            ;;
        run-full|full)
            run_full
            ;;
        all-quick)
            # One command to do everything quickly
            check_credits
            create_instance && \
            sleep 5 && \
            run_quick
            ;;
        help|*)
            echo "GCloud T4 Instance Manager for RL Token Compression"
            echo ""
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  create      - Create new T4 instance with everything set up"
            echo "  ssh         - Connect to instance via SSH"
            echo "  stop        - Stop instance (pause billing)"
            echo "  start       - Start stopped instance"
            echo "  delete      - Delete instance permanently"
            echo "  status      - Check instance and GPU status"
            echo "  run-quick   - Run quick training pipeline (10 min)"
            echo "  run-full    - Run full training pipeline (hours)"
            echo "  all-quick   - Create instance and run quick pipeline"
            echo ""
            echo "Examples:"
            echo "  $0 create           # Create instance"
            echo "  $0 ssh              # Connect to instance"
            echo "  $0 run-quick        # Run quick test"
            echo "  $0 stop             # Stop to save money"
            echo "  $0 all-quick        # Do everything in one command"
            echo ""
            echo "Set REPO_URL environment variable to use your own repo:"
            echo "  export REPO_URL=https://github.com/yourusername/rl-token-compression.git"
            ;;
    esac
}

# Run main function
main "$@"
#!/bin/bash

# SandboxFusion Server Management Script
# This script provides easy management of the SandboxFusion server

# Configuration
SANDBOX_DIR="SandboxFusion"
LOG_FILE="$SANDBOX_DIR/sandboxfusion.log"
PID_FILE="$SANDBOX_DIR/sandboxfusion.pid"
PORT=8080
CONDA_ENV="sandboxfusion"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  SandboxFusion Server Manager${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if server is running
is_server_running() {
    if pgrep -f "uvicorn sandbox.server.server" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to get server PID
get_server_pid() {
    pgrep -f "uvicorn sandbox.server.server"
}

# Function to check server health
check_server_health() {
    if curl -s http://localhost:$PORT/v1/ping > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to start the server
start_server() {
    print_status "Starting SandboxFusion server..."
    
    if is_server_running; then
        print_warning "Server is already running!"
        return 1
    fi
    
    # Check if conda environment exists
    if ! conda env list | grep -q "^$CONDA_ENV "; then
        print_error "Conda environment '$CONDA_ENV' not found!"
        print_status "Creating conda environment..."
        conda create -n $CONDA_ENV python=3.10 -y
    fi
    
    # Check if sandbox-runtime environment exists
    if ! conda env list | grep -q "^sandbox-runtime "; then
        print_status "Creating sandbox-runtime environment..."
        conda create -n sandbox-runtime python=3.10 -y
    fi
    
    # Change to SandboxFusion directory
    cd "$SANDBOX_DIR" || {
        print_error "Cannot change to SandboxFusion directory: $SANDBOX_DIR"
        return 1
    }
    
    # Start the server in background
    print_status "Starting server with nohup..."
    nohup bash -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && make run-online" > "$LOG_FILE" 2>&1 &
    
    # Save PID
    echo $! > "$PID_FILE"
    
    # Wait for server to start
    print_status "Waiting for server to start..."
    for i in {1..30}; do
        if check_server_health; then
            print_status "Server started successfully!"
            print_status "Server URL: http://localhost:$PORT"
            print_status "Log file: $LOG_FILE"
            print_status "PID file: $PID_FILE"
            return 0
        fi
        sleep 2
    done
    
    print_error "Server failed to start within 60 seconds"
    print_status "Check log file: $LOG_FILE"
    return 1
}

# Function to stop the server
stop_server() {
    print_status "Stopping SandboxFusion server..."
    
    if ! is_server_running; then
        print_warning "Server is not running!"
        return 0
    fi
    
    # Get PID
    local pid=$(get_server_pid)
    if [ -n "$pid" ]; then
        print_status "Found server process (PID: $pid)"
        
        # Try graceful shutdown first
        kill "$pid"
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! is_server_running; then
                print_status "Server stopped gracefully"
                rm -f "$PID_FILE"
                return 0
            fi
            sleep 1
        done
        
        # Force kill if graceful shutdown failed
        print_warning "Graceful shutdown failed, force killing..."
        kill -9 "$pid"
        sleep 2
        
        if ! is_server_running; then
            print_status "Server stopped forcefully"
            rm -f "$PID_FILE"
            return 0
        else
            print_error "Failed to stop server"
            return 1
        fi
    else
        print_error "Could not find server process"
        return 1
    fi
}

# Function to restart the server
restart_server() {
    print_status "Restarting SandboxFusion server..."
    stop_server
    sleep 2
    start_server
}

# Function to show server status
show_status() {
    print_status "Checking SandboxFusion server status..."
    
    if is_server_running; then
        local pid=$(get_server_pid)
        print_status "Server is RUNNING (PID: $pid)"
        
        if check_server_health; then
            print_status "Server is HEALTHY (responding to ping)"
            
            # Show recent log entries
            if [ -f "$LOG_FILE" ]; then
                echo
                print_status "Recent log entries:"
                tail -5 "$LOG_FILE" | sed 's/^/  /'
            fi
        else
            print_warning "Server is running but not responding to ping"
        fi
    else
        print_status "Server is NOT RUNNING"
    fi
    
    # Show port usage
    echo
    print_status "Port $PORT usage:"
    if lsof -i :$PORT > /dev/null 2>&1; then
        lsof -i :$PORT
    else
        print_status "No process using port $PORT"
    fi
}

# Function to test the server
test_server() {
    print_status "Testing SandboxFusion server..."
    
    if ! is_server_running; then
        print_error "Server is not running!"
        return 1
    fi
    
    if ! check_server_health; then
        print_error "Server is not responding to ping!"
        return 1
    fi
    
    # Run the test script
    if [ -f "test_real_sandboxfusion.py" ]; then
        print_status "Running test script..."
        python test_real_sandboxfusion.py
    else
        print_warning "Test script not found, running basic ping test..."
        response=$(curl -s http://localhost:$PORT/v1/ping)
        if [ "$response" = '"pong"' ]; then
            print_status "Basic test PASSED"
        else
            print_error "Basic test FAILED"
            return 1
        fi
    fi
}

# Function to show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        print_status "Showing SandboxFusion server logs:"
        echo
        tail -f "$LOG_FILE"
    else
        print_error "Log file not found: $LOG_FILE"
    fi
}

# Function to show help
show_help() {
    print_header
    echo "Usage: $0 {start|stop|restart|status|test|logs|help}"
    echo
    echo "Commands:"
    echo "  start   - Start the SandboxFusion server"
    echo "  stop    - Stop the SandboxFusion server"
    echo "  restart - Restart the SandboxFusion server"
    echo "  status  - Show server status and recent logs"
    echo "  test    - Test the server functionality"
    echo "  logs    - Show and follow server logs"
    echo "  help    - Show this help message"
    echo
    echo "Configuration:"
    echo "  Directory: $SANDBOX_DIR"
    echo "  Port: $PORT"
    echo "  Conda Environment: $CONDA_ENV"
    echo "  Log File: $LOG_FILE"
    echo "  PID File: $PID_FILE"
}

# Main script logic
case "$1" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        show_status
        ;;
    test)
        test_server
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo
        show_help
        exit 1
        ;;
esac

exit $? 
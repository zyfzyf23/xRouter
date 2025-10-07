# SandboxFusion Server Management - Quick Reference

## 🚀 Management Script

Use the `manage_sandboxfusion.sh` script to easily manage your SandboxFusion server:

```bash
./manage_sandboxfusion.sh [command]
```

## 📋 Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `start` | Start the SandboxFusion server | `./manage_sandboxfusion.sh start` |
| `stop` | Stop the SandboxFusion server | `./manage_sandboxfusion.sh stop` |
| `restart` | Restart the SandboxFusion server | `./manage_sandboxfusion.sh restart` |
| `status` | Show server status and recent logs | `./manage_sandboxfusion.sh status` |
| `test` | Test the server functionality | `./manage_sandboxfusion.sh test` |
| `logs` | Show and follow server logs | `./manage_sandboxfusion.sh logs` |
| `help` | Show help message | `./manage_sandboxfusion.sh help` |

## 🔧 Configuration

The script is configured with these settings:
- **Directory**: `SandboxFusion`
- **Port**: `8080`
- **Conda Environment**: `sandboxfusion`
- **Log File**: `SandboxFusion/sandboxfusion.log`
- **PID File**: `SandboxFusion/sandboxfusion.pid`

## 🎯 Quick Start

1. **Start the server**:
   ```bash
   ./manage_sandboxfusion.sh start
   ```

2. **Check status**:
   ```bash
   ./manage_sandboxfusion.sh status
   ```

3. **Test functionality**:
   ```bash
   ./manage_sandboxfusion.sh test
   ```

4. **Stop when done**:
   ```bash
   ./manage_sandboxfusion.sh stop
   ```

## 🌐 Server Access

Once started, the server is available at:
- **URL**: `http://localhost:8080`
- **Health Check**: `http://localhost:8080/v1/ping`
- **API Endpoint**: `http://localhost:8080/run_code`

## 📊 Status Indicators

The status command shows:
- ✅ **RUNNING** - Server is active and responding
- ✅ **HEALTHY** - Server responds to ping requests
- ❌ **NOT RUNNING** - Server is stopped
- ⚠️ **WARNING** - Server running but not responding

## 🔍 Troubleshooting

### Server won't start
- Check the log file: `tail -f SandboxFusion/sandboxfusion.log`
- Ensure conda environments exist: `conda env list`

### Server won't stop
- Force kill: `pkill -9 -f "uvicorn sandbox.server.server"`
- Check for zombie processes: `ps aux | grep uvicorn`

### Port conflicts
- Check port usage: `lsof -i :8080`
- Kill conflicting process: `kill $(lsof -ti:8080)`

## 📝 Manual Commands (if needed)

If you need to manage the server manually:

```bash
# Start manually
cd SandboxFusion
conda activate sandboxfusion
make run-online

# Stop manually
pkill -f "uvicorn sandbox.server.server"

# Check if running
ps aux | grep "uvicorn sandbox.server.server" | grep -v grep

# Test manually
curl http://localhost:8080/v1/ping
```

## 🎉 Success!

Your SandboxFusion server is now fully managed and ready for use! 🚀 
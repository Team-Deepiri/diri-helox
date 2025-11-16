# Scripts Directory

This directory contains all utility scripts for Deepiri. Scripts are organized by purpose.

## 🚀 Quick Reference

### Main Rebuild Scripts (in root for easy access)
- **`../rebuild.sh`** / **`../rebuild.ps1`** - Main rebuild scripts (clean rebuild, removes old images)
  - Use these when you need to rebuild after code changes
  - Located in root for easy access: `./rebuild.sh` or `.\rebuild.ps1`

### Cleanup Scripts
- **`stop-and-cleanup.sh`** / **`stop-and-cleanup.ps1`** - Stop containers and clean Docker resources
- **`cleanup-and-compact.ps1`** - Full cleanup including WSL2 disk compaction (Windows, requires Admin)
- **`run-cleanup-direct.ps1`** - PowerShell launcher for cleanup scripts
- **`run-cleanup-as-admin.bat`** - Batch file to run cleanup as Administrator (Windows)

### Development Scripts
- **`dev-docker.sh`** - Development Docker utilities
- **`dev-start.js`** - Development server starter
- **`run.py`** - Python utility runner
- **`utils_docker.py`** - Docker utility functions

### Other Scripts
- **`setup.sh`** - Initial project setup
- **`docker-cleanup.sh`** - Docker cleanup utilities
- **`test-runner.sh`** - Test execution script
- **`fix-dependencies.sh`** - Dependency fixer
- **`mongo-backup.sh`** / **`mongo-restore.sh`** - MongoDB backup/restore
- **`compact-wsl-disk.sh`** / **`compact-wsl-disk.ps1`** - WSL2 disk compaction

### Archive
- **`archive/`** - Old/legacy scripts (kept for reference)

---

## 📖 Detailed Script Documentation

### Rebuild Scripts

#### `../rebuild.sh` / `../rebuild.ps1` (Main - in root)
**Purpose:** Clean rebuild of all Docker services

**Usage:**
```bash
# Linux/Mac
./rebuild.sh

# Windows PowerShell
.\rebuild.ps1

# Specify different compose file
./rebuild.sh docker-compose.yml
.\rebuild.ps1 -ComposeFile docker-compose.yml
```

**What it does:**
1. Stops all containers
2. Removes old images (prevents storage bloat!)
3. Cleans build cache
4. Rebuilds everything fresh (no cache)
5. Starts all services

**When to use:**
- After code changes
- When you want fresh images
- When Docker storage is getting full

**Note:** Normal `docker compose up` does NOT rebuild - use these scripts when you need to rebuild.

---

### Cleanup Scripts

#### `stop-and-cleanup.sh` / `stop-and-cleanup.ps1`
**Purpose:** Stop containers and clean Docker resources (images, volumes, cache)

**Usage:**
```bash
# Linux/Mac
./scripts/stop-and-cleanup.sh

# Windows PowerShell
.\scripts\stop-and-cleanup.ps1
```

**What it does:**
- Stops all Deepiri containers
- Removes unused images
- Prunes volumes
- Cleans build cache

**When to use:**
- When you want to clean up without rebuilding
- To free Docker storage space

---

#### `cleanup-and-compact.ps1` (Windows only)
**Purpose:** Full cleanup including WSL2 disk compaction

**Usage:**
```powershell
# Requires Administrator privileges
.\scripts\cleanup-and-compact.ps1
```

**What it does:**
1. Docker cleanup (images, volumes, cache)
2. Shuts down WSL2 safely
3. Compacts Ubuntu VHDX file
4. Restarts WSL2
5. Shows space reclaimed

**When to use:**
- When WSL2 disk is very large (50GB+)
- When you need maximum space reclamation
- Requires Administrator privileges

**Helper scripts:**
- `run-cleanup-as-admin.bat` - Right-click → Run as Administrator
- `run-cleanup-direct.ps1` - PowerShell launcher

---

### Development Scripts

#### `dev-docker.sh`
**Purpose:** Development Docker utilities

**Usage:**
```bash
./scripts/dev-docker.sh [command]
```

#### `dev-start.js`
**Purpose:** Development server starter

**Usage:**
```bash
node scripts/dev-start.js
```

#### `run.py`
**Purpose:** Python utility runner

**Usage:**
```bash
python scripts/run.py [options]
```

---

### Database Scripts

#### `mongo-backup.sh`
**Purpose:** Backup MongoDB database

**Usage:**
```bash
./scripts/mongo-backup.sh
```

#### `mongo-restore.sh`
**Purpose:** Restore MongoDB database from backup

**Usage:**
```bash
./scripts/mongo-restore.sh [backup-file]
```

---

### Setup Scripts

#### `setup.sh`
**Purpose:** Initial project setup

**Usage:**
```bash
./scripts/setup.sh
```

**What it does:**
- Creates necessary directories
- Sets up environment files
- Installs dependencies
- Configures Docker

---

## 🔧 Script Organization

```
scripts/
├── README.md                    # This file
├── cleanup-*.sh/ps1          # Cleanup scripts
├── dev-*.sh/js                # Development utilities
├── mongo-*.sh                 # Database scripts
├── setup.sh                   # Initial setup
├── test-runner.sh             # Testing utilities
├── fix-dependencies.sh        # Dependency management
├── compact-wsl-disk.*         # WSL2 disk management
└── archive/                   # Old/legacy scripts
```

---

## 📝 Best Practices

1. **Use rebuild scripts** (`../rebuild.sh` / `../rebuild.ps1`) for rebuilding
2. **Use cleanup scripts** when you need to free space
3. **Check script help** - Most scripts have `--help` or usage info
4. **Run as Administrator** when required (Windows cleanup scripts)

---

## 🆘 Troubleshooting

### Script won't run (PowerShell)
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\script-name.ps1
```

### Script opens in Notepad
- Use the `.bat` helper files
- Or run from PowerShell directly

### Permission denied (Linux/Mac)
```bash
chmod +x scripts/script-name.sh
```

---

## 📚 Related Documentation

- **[../docs/DOCKER-IMAGE-CLEANSING-COMMANDS.md](../docs/DOCKER-IMAGE-CLEANSING-COMMANDS.md)** - Complete Docker cleanup guide
- **[../docs/MAKEFILE-EXPLANATION.md](../docs/MAKEFILE-EXPLANATION.md)** - Makefile usage (alternative to scripts)
- **[../README.md](../README.md)** - Main project README

---

**Questions?** Check the main [README.md](../README.md) or [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md)


# Deepiri Docker Stop and Cleanup Script (PowerShell)
# This script stops all containers and cleans up Docker resources
# Usage: .\stop-and-cleanup.ps1 [-KeepVolumes] [-KeepImages]

param(
    [switch]$KeepVolumes,
    [switch]$KeepImages
)

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput Cyan "========================================"
Write-ColorOutput Cyan "Deepiri Docker Stop and Cleanup Script"
Write-ColorOutput Cyan "========================================"
Write-Output ""

# Function to check if Docker is running
function Test-Docker {
    try {
        docker info | Out-Null
        return $true
    } catch {
        Write-ColorOutput Red "Error: Docker is not running or not accessible"
        exit 1
    }
}

# Function to stop all containers
function Stop-Containers {
    Write-ColorOutput Yellow "Stopping all Deepiri containers..."
    
    $containers = docker ps -a --filter "name=deepiri" --format "{{.Names}}" 2>$null
    
    if ([string]::IsNullOrWhiteSpace($containers)) {
        Write-ColorOutput Green "No Deepiri containers found"
    } else {
        $containerList = $containers -split "`n" | Where-Object { $_ -and $_.Trim() }
        foreach ($container in $containerList) {
            if ($container) {
                Write-Output "  Stopping: $container"
                docker stop $container 2>$null | Out-Null
            }
        }
        
        # Also stop any containers started by docker-compose
        Write-ColorOutput Yellow "Stopping docker-compose services..."
        docker-compose -f docker-compose.yml down 2>$null | Out-Null
        docker-compose -f docker-compose.dev.yml down 2>$null | Out-Null
        docker-compose -f docker-compose.microservices.yml down 2>$null | Out-Null
        docker-compose -f docker-compose.enhanced.yml down 2>$null | Out-Null
        
        Write-ColorOutput Green "✓ All containers stopped"
    }
}

# Function to remove containers
function Remove-Containers {
    Write-ColorOutput Yellow "Removing all Deepiri containers..."
    
    $containers = docker ps -a --filter "name=deepiri" --format "{{.Names}}" 2>$null
    
    if ([string]::IsNullOrWhiteSpace($containers)) {
        Write-ColorOutput Green "No Deepiri containers to remove"
    } else {
        $containerList = $containers -split "`n" | Where-Object { $_ -and $_.Trim() }
        foreach ($container in $containerList) {
            if ($container) {
                Write-Output "  Removing: $container"
                docker rm -f $container 2>$null | Out-Null
            }
        }
        
        Write-ColorOutput Green "✓ All containers removed"
    }
}

# Function to remove images
function Remove-Images {
    if ($KeepImages) {
        Write-ColorOutput Yellow "Skipping image removal (-KeepImages flag set)"
        return
    }
    
    Write-ColorOutput Yellow "Removing Deepiri images..."
    
    $images = docker images --filter "reference=*deepiri*" --format "{{.Repository}}:{{.Tag}}" 2>$null
    
    if ([string]::IsNullOrWhiteSpace($images)) {
        Write-ColorOutput Green "No Deepiri images found"
    } else {
        $imageList = $images -split "`n" | Where-Object { $_ -and $_.Trim() }
        foreach ($image in $imageList) {
            if ($image) {
                Write-Output "  Removing: $image"
                docker rmi -f $image 2>$null | Out-Null
            }
        }
        
        Write-ColorOutput Green "✓ Images removed"
    }
    
    # Also remove dangling images
    Write-ColorOutput Yellow "Removing dangling images..."
    $dangling = docker images -f "dangling=true" -q 2>$null
    if ($dangling) {
        docker rmi -f $dangling 2>$null | Out-Null
        Write-ColorOutput Green "✓ Dangling images removed"
    } else {
        Write-ColorOutput Green "No dangling images found"
    }
}

# Function to remove volumes
function Remove-Volumes {
    if ($KeepVolumes) {
        Write-ColorOutput Yellow "Skipping volume removal (-KeepVolumes flag set)"
        return
    }
    
    Write-ColorOutput Yellow "Removing unused volumes..."
    
    $volumes = docker volume ls --filter "name=deepiri" --format "{{.Name}}" 2>$null
    
    if ([string]::IsNullOrWhiteSpace($volumes)) {
        Write-ColorOutput Green "No Deepiri volumes found"
    } else {
        $volumeList = $volumes -split "`n" | Where-Object { $_ -and $_.Trim() }
        foreach ($volume in $volumeList) {
            if ($volume) {
                Write-Output "  Removing: $volume"
                docker volume rm $volume 2>$null | Out-Null
            }
        }
        
        Write-ColorOutput Green "✓ Volumes removed"
    }
    
    # Remove all unused volumes
    Write-ColorOutput Yellow "Removing all unused volumes..."
    docker volume prune -f 2>$null | Out-Null
    Write-ColorOutput Green "✓ Unused volumes pruned"
}

# Function to clean build cache
function Clear-BuildCache {
    Write-ColorOutput Yellow "Cleaning Docker build cache..."
    docker builder prune -af 2>$null | Out-Null
    Write-ColorOutput Green "✓ Build cache cleaned"
}

# Function to remove networks
function Remove-Networks {
    Write-ColorOutput Yellow "Removing Deepiri networks..."
    
    $networks = docker network ls --filter "name=deepiri" --format "{{.Name}}" 2>$null
    
    if ([string]::IsNullOrWhiteSpace($networks)) {
        Write-ColorOutput Green "No Deepiri networks found"
    } else {
        $networkList = $networks -split "`n" | Where-Object { $_ -and $_.Trim() }
        foreach ($network in $networkList) {
            if ($network -and $network -ne "deepiri-network" -and $network -ne "deepiri-dev-network") {
                Write-Output "  Removing: $network"
                docker network rm $network 2>$null | Out-Null
            }
        }
        
        # Remove the main networks if no containers are using them
        foreach ($network in @("deepiri-network", "deepiri-dev-network")) {
            try {
                $networkInfo = docker network inspect $network 2>$null
                if ($networkInfo) {
                    $containers = docker network inspect $network --format '{{len .Containers}}' 2>$null
                    if ($containers -eq "0") {
                        Write-Output "  Removing: $network"
                        docker network rm $network 2>$null | Out-Null
                    }
                }
            } catch {
                # Network doesn't exist, skip
            }
        }
        
        Write-ColorOutput Green "✓ Networks removed"
    }
}

# Function to show disk usage
function Show-DiskUsage {
    Write-Output ""
    Write-ColorOutput Cyan "Docker Disk Usage:"
    docker system df
    Write-Output ""
}

# Main execution
function Main {
    Test-Docker
    
    Write-ColorOutput Yellow "Current Docker disk usage:"
    Show-DiskUsage
    
    # Confirm before proceeding
    if (-not $KeepVolumes -or -not $KeepImages) {
        Write-ColorOutput Red "WARNING: This will remove Docker resources!"
        if (-not $KeepVolumes) {
            Write-ColorOutput Red "  - All volumes will be removed"
        }
        if (-not $KeepImages) {
            Write-ColorOutput Red "  - All images will be removed"
        }
        Write-ColorOutput Red "  - Build cache will be cleared"
        Write-Output ""
        $confirmation = Read-Host "Continue? (y/N)"
        if ($confirmation -ne "y" -and $confirmation -ne "Y") {
            Write-ColorOutput Yellow "Aborted"
            exit 0
        }
    }
    
    Write-Output ""
    
    # Execute cleanup steps
    Stop-Containers
    Remove-Containers
    Remove-Images
    Remove-Volumes
    Clear-BuildCache
    Remove-Networks
    
    Write-Output ""
    Write-ColorOutput Green "========================================"
    Write-ColorOutput Green "Cleanup Complete!"
    Write-ColorOutput Green "========================================"
    Write-Output ""
    
    Write-ColorOutput Yellow "Final Docker disk usage:"
    Show-DiskUsage
    
    Write-ColorOutput Green "All Deepiri Docker resources have been cleaned up!"
}

# Run main function
Main


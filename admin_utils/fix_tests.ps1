<#
.SYNOPSIS
    Smartly moves files from 'tests' to 'src/tests'.
    
    BEHAVIOR:
    - Default: DRY RUN (Safe mode, only prints what would happen).
    - With -Execute: Actually performs the moves and deletes.

    LOGIC:
    - Identical Content? -> Deletes the duplicate in 'tests/'.
    - Unique File?       -> Moves it to 'src/tests/' (using git mv).
    - Different Content? -> SKIPS it (Conflict protection).

.PARAMETER SourceDir
    Source directory (default: ../tests)

.PARAMETER DestDir
    Destination directory (default: ../src/tests)

.PARAMETER Execute
    Required to actually perform changes.

.EXAMPLE
    .\Move-TestsToSrc.ps1              # Dry Run
    .\Move-TestsToSrc.ps1 -Execute     # Real Run
#>

param (
    [string]$SourceDir = "../tests",
    [string]$DestDir = "../src/tests",
    [switch]$Execute
)

# --- CONFIGURATION & SETUP ---
$IsDryRun = -not $Execute

if ($IsDryRun) {
    Write-Host ">>> DRY RUN MODE <<<" -ForegroundColor Yellow
    Write-Host "No files will be touched. Use -Execute to apply changes." -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host ">>> EXECUTION MODE <<<" -ForegroundColor Red
    Write-Host "Applying changes to file system..." -ForegroundColor Gray
    Write-Host ""
}

# Resolve paths
if (-not (Test-Path $SourceDir)) {
    Write-Warning "Source directory '$SourceDir' not found."
    exit
}

$absSource = Convert-Path $SourceDir
$absDest = $null

# Handle Destination
if (Test-Path $DestDir) {
    $absDest = Convert-Path $DestDir
} elseif (-not $IsDryRun) {
    Write-Host "Creating destination '$DestDir'..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
    $absDest = Convert-Path $DestDir
} else {
    $absDest = "[Target Directory]"
}

Write-Host "Scanning '$SourceDir'..." -ForegroundColor Cyan

$files = Get-ChildItem -Path $SourceDir -Recurse -File

$stats = @{
    ToMove    = 0
    ToDelete  = 0
    ToSkip    = 0
}

# --- PROCESS FILES ---

foreach ($file in $files) {
    # Calculate relative path (e.g. "unit\test_foo.py")
    # Using Substring ensures we get the path relative to the SourceDir root
    $relativePath = $file.FullName.Substring($absSource.Length + 1)
    
    # Destination path
    $destPath = Join-Path $DestDir $relativePath
    $parentDir = Split-Path $destPath -Parent

    # Check if destination exists
    if (Test-Path $destPath) {
        # --- CONFLICT CHECK ---
        $srcHash = Get-FileHash $file.FullName -Algorithm MD5
        $destHash = Get-FileHash $destPath -Algorithm MD5

        if ($srcHash.Hash -eq $destHash.Hash) {
            # CASE: IDENTICAL (Duplicate) -> DELETE SOURCE
            if ($IsDryRun) {
                Write-Host "[Plan] DELETE DUPLICATE: $relativePath" -ForegroundColor DarkGray
            } else {
                Write-Host "DELETE DUPLICATE: $relativePath" -ForegroundColor Gray
                # Use git rm if tracked, fallback to file delete
                git rm --quiet "$($file.FullName)" 2>$null
                if (Test-Path $file.FullName) {
                    Remove-Item $file.FullName -Force
                }
            }
            $stats.ToDelete++
        } else {
            # CASE: DIFFERENT -> SKIP (Conflict)
            Write-Host "[Skip] CONFLICT: $relativePath" -ForegroundColor Red
            $stats.ToSkip++
        }
    } else {
        # --- SAFE MOVE ---
        if ($IsDryRun) {
            Write-Host "[Plan] GIT MOVE: $relativePath" -ForegroundColor Green
        } else {
            # Ensure parent dir exists
            if (-not (Test-Path $parentDir)) {
                New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
            }

            Write-Host "GIT MOVE: $relativePath" -ForegroundColor Green
            git mv "$($file.FullName)" "$destPath"
            
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Git move failed for $relativePath"
            }
        }
        $stats.ToMove++
    }
}

# --- CLEANUP EMPTY FOLDERS ---
if ($IsDryRun) {
    Write-Host ""
    Write-Host "[Plan] Cleanup empty folders in '$SourceDir'" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "Cleaning up empty folders in '$SourceDir'..." -ForegroundColor Cyan
    
    # Sort Descending by length deletes children before parents
    Get-ChildItem -Path $SourceDir -Recurse -Directory | 
        Sort-Object FullName -Descending | 
        ForEach-Object {
            if ((Get-ChildItem $_.FullName).Count -eq 0) {
                Remove-Item $_.FullName
            }
        }
    
    # Try removing root
    if ((Get-ChildItem $SourceDir).Count -eq 0) {
        Remove-Item $SourceDir
        Write-Host "Removed root '$SourceDir'" -ForegroundColor Cyan
    } else {
        Write-Warning "Root '$SourceDir' still contains files (conflicts)."
    }
}

# --- SUMMARY ---
Write-Host ""
Write-Host "--------------------------------------------------"
Write-Host "Run Summary"
Write-Host "--------------------------------------------------"
Write-Host "  Mode      : $(if($IsDryRun){'DRY RUN'}else{'EXECUTE'})"
Write-Host "  Moves     : $($stats.ToMove)"
Write-Host "  Deletes   : $($stats.ToDelete) (Identical Duplicates)"
Write-Host "  Conflicts : $($stats.ToSkip) (Manual Resolution Needed)"
Write-Host "--------------------------------------------------"
if ($stats.ToSkip -gt 0) {
    Write-Warning "You have $($stats.ToSkip) files that differ in both locations."
    Write-Warning "Please inspect these files manually."
}
if ($IsDryRun) {
    Write-Host "Run with -Execute to apply these changes." -ForegroundColor Yellow
}
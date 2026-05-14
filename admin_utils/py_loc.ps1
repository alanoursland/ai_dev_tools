<#
.SYNOPSIS
    Counts lines of code in .py files within a specific directory.

.DESCRIPTION
    Recursively scans the provided directory for Python files.
    Calculates total lines and lists the largest files.

.PARAMETER Path
    The folder path to scan.

.EXAMPLE
    .\Count-PyLOC.ps1 -Path "src"
#>

param (
    [Parameter(Mandatory=$true)]
    [string]$Path
)

# Resolve full path to ensure it looks nice in output
$fullPath = Convert-Path $Path 2>$null

if (-not $fullPath -or -not (Test-Path $fullPath)) {
    Write-Error "Directory not found: $Path"
    exit
}

Write-Host "Scanning '$fullPath' for Python files..." -ForegroundColor Cyan

# Find all .py files recursively
$files = Get-ChildItem -Path $fullPath -Recurse -Filter "*.py"

if ($files.Count -eq 0) {
    Write-Warning "No Python files found in this directory."
    exit
}

$totalLines = 0
$fileStats = @()

foreach ($file in $files) {
    # Force reading as text to avoid issues, Measure-Object counts lines
    $lineCount = (Get-Content $file.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    
    $totalLines += $lineCount
    
    $fileStats += [PSCustomObject]@{
        Name  = $file.Name
        Path  = $file.FullName.Replace($fullPath + "\", "") # Relative path for cleaner display
        Lines = $lineCount
    }
}

Write-Host "--------------------------------------------------"
Write-Host "Summary for '$fullPath'"
Write-Host "Total Files : $($files.Count)"
Write-Host "Total LOC   : $totalLines"
Write-Host "--------------------------------------------------"
Write-Host ""

Write-Host "TOP 20 LARGEST FILES:" -ForegroundColor Yellow
$fileStats | 
    Sort-Object Lines -Descending | 
    Select-Object -First 20 | 
    Format-Table -AutoSize -Property Lines, Path
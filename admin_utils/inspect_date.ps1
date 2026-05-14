<#
.SYNOPSIS
    Analyzes git changes for a specific date.
    
    UPDATED: 
    - Lists COMMITS for the day (Context).
    - Lists DETECTED RENAMES explicitly.
    - Tracks both Additions AND Removals per file.
    - NEW: Shows FILE COUNTS per directory (helpful for AI generation analysis).
    - Lists top 50 files.

.PARAMETER Date
    YYYY-MM-DD date to analyze.

.PARAMETER SimilarityThreshold
    Percentage (0-100). Lower this if Git isn't detecting your moves.
    Default: 20 (Aggressive detection). Standard Git default is 50.

.PARAMETER ExcludeRenames
    If set, completely hides files that Git identifies as Renames.

.EXAMPLE
    .\Get-GitDayDetails.ps1 -Date "2025-11-22" -SimilarityThreshold 10
#>

param (
    [Parameter(Mandatory=$true)]
    [string]$Date,

    [int]$SimilarityThreshold = 20,

    [switch]$ExcludeRenames
)

try {
    $parsedDate = Get-Date $Date
    $dateStr = $parsedDate.ToString("yyyy-MM-dd")
} catch {
    Write-Error "Invalid date format. Please use YYYY-MM-DD."
    exit
}

Write-Host "Analyzing git activity for $dateStr" -ForegroundColor Cyan
Write-Host " - Rename Detection Threshold: $SimilarityThreshold%" -ForegroundColor DarkGray

# --- STEP 1: PRINT COMMITS ---
Write-Host ""
Write-Host "COMMITS ON THIS DAY:" -ForegroundColor Yellow
$commits = git log --after="$dateStr 00:00" --before="$dateStr 23:59" --pretty=format:"%h - %s (%an)" 2>$null
if ($commits) {
    $commits | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Warning "No commits found."
    exit
}
Write-Host ""

# --- STEP 2: ANALYZE STATS ---

# Build Git Command
$gitArgs = @("log", "--after=$dateStr 00:00", "--before=$dateStr 23:59", "--numstat", "-M$($SimilarityThreshold)%", "-C$($SimilarityThreshold)%")

if ($ExcludeRenames) {
    $gitArgs += "--diff-filter=r"
}

$gitArgs += "--format="

$logOutput = git @gitArgs 2>$null

# Store objects with {Added, Removed}
$fileStats = @{}
$dirStats = @{} 
$detectedRenames = @()
$totalLinesAdded = 0
$totalLinesRemoved = 0
$totalFiles = 0
$pyLines = 0

foreach ($line in $logOutput) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }

    $parts = $line -split "\t"
    if ($parts.Count -lt 3) { continue }

    $addedStr = $parts[0]
    $removedStr = $parts[1]
    $filePath = $parts[2]

    # Skip binary
    if ($addedStr -eq "-" -or $removedStr -eq "-") { continue }

    # Handle quoted paths
    if ($filePath.StartsWith('"') -and $filePath.EndsWith('"')) {
        $filePath = $filePath.Substring(1, $filePath.Length - 2)
    }

    # --- CAPTURE RENAMES ---
    if ($filePath -match '=>') {
        $detectedRenames += $filePath
    }

    # --- RESOLVE PATHS FOR STATS ---
    # 1. Handle brace expansion: "src/{old => new}/file.py" -> "src/new/file.py"
    if ($filePath -match '\{.*? => (.*?)\}') {
        $filePath = $filePath -replace '\{.*? => (.*?)\}', '$1'
    }
    # 2. Handle full path renames: "old/file.py => new/file.py" -> "new/file.py"
    if ($filePath -match '.*? => (.*)') {
        $filePath = $filePath -replace '.*? => (.*)', '$1'
    }
    # -----------------------

    [int]$added = $addedStr
    [int]$removed = $removedStr
    
    # Update Totals
    $totalLinesAdded += $added
    $totalLinesRemoved += $removed
    $totalFiles++

    if ($filePath -match '\.py$') {
        $pyLines += $added
    }

    # Update File Stats
    if (-not $fileStats.ContainsKey($filePath)) {
        $fileStats[$filePath] = [PSCustomObject]@{ Added = 0; Removed = 0 }
    }
    $fileStats[$filePath].Added += $added
    $fileStats[$filePath].Removed += $removed

    # Update Directory Stats
    $parentDir = "(root)"
    if ($filePath -match '^(.*)[/\\]') { 
        $parentDir = $Matches[1]
    }
    
    if (-not $dirStats.ContainsKey($parentDir)) {
        # Initialize object with counts
        $dirStats[$parentDir] = [PSCustomObject]@{ LinesAdded = 0; FileCount = 0 }
    }
    $dirStats[$parentDir].LinesAdded += $added
    $dirStats[$parentDir].FileCount += 1
}

Write-Host "--------------------------------------------------"
Write-Host "Summary for $dateStr"
Write-Host "Total Lines Added   : $totalLinesAdded"
Write-Host "Total Lines Removed : $totalLinesRemoved"
Write-Host "Total Python Added  : $pyLines"
Write-Host "Total Files Touched : $totalFiles"
Write-Host "--------------------------------------------------"
Write-Host ""

if ($detectedRenames.Count -gt 0) {
    Write-Host "DETECTED MOVES/RENAMES ($($detectedRenames.Count)):" -ForegroundColor Cyan
    $detectedRenames | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" }
    if ($detectedRenames.Count -gt 10) { Write-Host "  ... and $($detectedRenames.Count - 10) more." }
    Write-Host ""
} else {
    Write-Host "NO MOVES DETECTED." -ForegroundColor Red
    Write-Host ""
}

Write-Host "TOP 10 DIRECTORIES BY VOLUME (Added Only):" -ForegroundColor Yellow
$dirStats.GetEnumerator() | 
    Sort-Object -Property @{Expression={$_.Value.LinesAdded}; Descending=$true} | 
    Select-Object -First 10 |
    Format-Table -AutoSize -Property `
        @{N="Lines Added"; E={$_.Value.LinesAdded}}, `
        @{N="Files"; E={$_.Value.FileCount}}, `
        @{N="Directory"; E={$_.Name}}

Write-Host "TOP 50 INDIVIDUAL FILES (Sorted by Additions):" -ForegroundColor Yellow
$fileStats.GetEnumerator() | 
    Select-Object @{N="File"; E={$_.Key}}, @{N="Added"; E={$_.Value.Added}}, @{N="Removed"; E={$_.Value.Removed}} |
    Sort-Object Added -Descending | 
    Select-Object -First 50 |
    Format-Table -AutoSize
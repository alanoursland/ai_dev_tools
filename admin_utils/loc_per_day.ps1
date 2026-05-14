<#
.SYNOPSIS
    Calculates git additions and removals per day, categorized by file type.

.DESCRIPTION
    Analyzes the git log for the specified number of past days.
    Groups statistics by PY, MD, and Other extensions.
    Ignores binary file changes.

.PARAMETER Days
    Number of days back to analyze. Default is 7.

.EXAMPLE
    .\Get-GitStats.ps1 -Days 14
#>

param (
    [int]$Days = 7
)

# Ensure we are in a git repository
if (-not (Test-Path .git) -and -not (git rev-parse --git-dir 2>$null)) {
    Write-Error "Current directory is not a git repository."
    exit
}

Write-Host "Analyzing git history for the last $Days days..." -ForegroundColor Cyan

# Get git log with numstat
# Format: "DATE:YYYY-MM-DD" followed by file stats lines "added removed filename"
$dateCommand = Get-Date (Get-Date).AddDays(-$Days) -Format "yyyy-MM-dd"
$gitArgs = @("log", "--since=$dateCommand", "--numstat", "--date=short", "--format=DATE:%ad")

$logOutput = git @gitArgs 2>$null

if (-not $logOutput) {
    Write-Warning "No commits found in the last $Days days."
    exit
}

# Data structure to hold aggregated stats
# Key: Date String, Value: Object with counts
$stats = [ordered]@{}

$currentDate = $null

foreach ($line in $logOutput) {
    # Check for Date marker
    if ($line -match "^DATE:(\d{4}-\d{2}-\d{2})") {
        $currentDate = $Matches[1]
        
        if (-not $stats.Contains($currentDate)) {
            $stats[$currentDate] = [PSCustomObject]@{
                Date      = $currentDate
                PY_Add    = 0
                PY_Del    = 0
                MD_Add    = 0
                MD_Del    = 0
                Other_Add = 0
                Other_Del = 0
            }
        }
        continue
    }

    if ([string]::IsNullOrWhiteSpace($line)) { continue }

    $parts = $line -split "\t"

    if ($parts.Count -lt 3) { continue }

    $addedStr = $parts[0]
    $removedStr = $parts[1]
    $filePath = $parts[2]

    # Skip binary files
    if ($addedStr -eq "-" -or $removedStr -eq "-") { continue }

    [int]$added = $addedStr
    [int]$removed = $removedStr

    # --- FIX START ---
    # Git puts quotes around paths with spaces/special chars. Remove them.
    if ($filePath.StartsWith('"') -and $filePath.EndsWith('"')) {
        $filePath = $filePath.Substring(1, $filePath.Length - 2)
    }

    # Use Regex to find extension instead of [System.IO.Path]
    # matches dot followed by alphanumeric chars at end of string
    $ext = ""
    if ($filePath -match '\.([a-zA-Z0-9]+)$') {
        $ext = "." + $Matches[1].ToLower()
    }
    # --- FIX END ---

    $entry = $stats[$currentDate]

    switch ($ext) {
        ".py" {
            $entry.PY_Add += $added
            $entry.PY_Del += $removed
        }
        ".md" {
            $entry.MD_Add += $added
            $entry.MD_Del += $removed
        }
        Default {
            $entry.Other_Add += $added
            $entry.Other_Del += $removed
        }
    }
}

$stats.Values | Sort-Object Date | Format-Table -AutoSize -Property `
    @{N="Date"; E={$_.Date}}, 
    @{N="PY +"; E={$_.PY_Add}; Align="Right"}, 
    @{N="PY -"; E={$_.PY_Del}; Align="Right"}, 
    @{N="MD +"; E={$_.MD_Add}; Align="Right"}, 
    @{N="MD -"; E={$_.MD_Del}; Align="Right"}, 
    @{N="Other +"; E={$_.Other_Add}; Align="Right"}, 
    @{N="Other -"; E={$_.Other_Del}; Align="Right"}

Write-Host "Done." -ForegroundColor Cyan
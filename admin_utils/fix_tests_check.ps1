<#
.SYNOPSIS
    Checks for file conflicts before moving 'tests' to 'src/tests'.

.DESCRIPTION
    Scans the root 'tests' folder.
    Calculates where each file WOULD go in 'src/tests'.
    Checks if a file already exists at that destination.
    If it exists, compares MD5 hashes to check for identical content.

.EXAMPLE
    .\Check-MoveConflicts.ps1
#>

$sourceDir = "../tests"
$destDir = "../src/tests"

if (-not (Test-Path $sourceDir)) {
    Write-Warning "Root '$sourceDir' directory not found."
    exit
}

Write-Host "Checking for conflicts between '$sourceDir' and '$destDir'..." -ForegroundColor Cyan

$files = Get-ChildItem -Path $sourceDir -Recurse -File
$conflicts = @()
$safeMoves = 0
$identical = 0

foreach ($file in $files) {
    # Calculate relative path (e.g. "unit\test_foo.py")
    $relativePath = $file.FullName.Substring((Get-Item $sourceDir).FullName.Length + 1)
    $destPath = Join-Path $destDir $relativePath

    if (Test-Path $destPath) {
        # Conflict found. Check content.
        $srcHash = Get-FileHash $file.FullName -Algorithm MD5
        $destHash = Get-FileHash $destPath -Algorithm MD5

        if ($srcHash.Hash -eq $destHash.Hash) {
            Write-Host "EXISTS (IDENTICAL): $relativePath" -ForegroundColor Gray
            $identical++
        } else {
            Write-Host "CONFLICT (DIFFERENT): $relativePath" -ForegroundColor Red
            $conflicts += [PSCustomObject]@{
                RelativePath = $relativePath
                SourceSize   = "{0:N2} KB" -f ($file.Length / 1KB)
                DestSize     = "{0:N2} KB" -f ((Get-Item $destPath).Length / 1KB)
            }
        }
    } else {
        # Write-Host "Safe: $relativePath" -ForegroundColor DarkGray
        $safeMoves++
    }
}

Write-Host ""
Write-Host "--------------------------------------------------"
Write-Host "Summary"
Write-Host "--------------------------------------------------"
Write-Host "  Safe to move : $safeMoves files"
Write-Host "  Identical    : $identical files (Target exists but content is same)"
Write-Host "  Conflicts    : $($conflicts.Count) files (Target exists AND content differs)"
Write-Host "--------------------------------------------------"
Write-Host ""

if ($conflicts.Count -gt 0) {
    Write-Host "CRITICAL CONFLICTS (Content Differs):" -ForegroundColor Red
    $conflicts | Format-Table -AutoSize
    
    Write-Warning "DO NOT run the move script yet."
    Write-Warning "You must manually resolve these conflicts (rename or delete one version)."
} else {
    Write-Host "No content conflicts found." -ForegroundColor Green
    if ($identical -gt 0) {
        Write-Host "Note: $identical files are identical duplicates. The move script will likely fail on these."
        Write-Host "You should delete the copy in 'tests/' for these specific files before moving."
    } else {
        Write-Host "You are safe to run the move script." -ForegroundColor Green
    }
}
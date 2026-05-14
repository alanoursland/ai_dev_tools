param(
    [switch]$pr,     # If set, create PRs for branches ahead of default
    [switch]$merge   # If set, merge open PRs that are clean/mergeable
)

# Get repo info (name + default branch) for the current directory
$repoInfoJson = gh repo view --json nameWithOwner,defaultBranchRef
$repoInfo = $repoInfoJson | ConvertFrom-Json

$repo = $repoInfo.nameWithOwner
$defaultBranch = $repoInfo.defaultBranchRef.name

Write-Host "Repository: $repo"
Write-Host "Default branch: $defaultBranch"
Write-Host ""

# 1) Get all branches on the repo
Write-Host "Fetching remote branches..."
$branchesJson = gh api "repos/$repo/branches" --paginate
$branches = $branchesJson | ConvertFrom-Json
$branchNames = $branches | ForEach-Object { $_.name }

# 2) Get all OPEN PRs and their head branches
Write-Host "Fetching open pull requests..."
$openPrsJson = gh pr list --state open --json headRefName,number,title
$openPrs = $openPrsJson | ConvertFrom-Json

$prBranchNames = @()
if ($openPrs) {
    $prBranchNames = $openPrs | ForEach-Object { $_.headRefName }
}

# 3) Branches that are NOT default and NOT used in an open PR
$branchesNeedingPr = $branchNames | Where-Object {
    $_ -ne $defaultBranch -and $prBranchNames -notcontains $_
}

Write-Host ""
Write-Host "== Branches on $repo that do NOT have an open PR =="
if (-not $branchesNeedingPr) {
    Write-Host "(none 🎉)"
} else {
    foreach ($b in $branchesNeedingPr) {
        Write-Host " - $b"
    }
}

# 4) For each such branch, get ahead/behind relative to default branch
$results = @()

foreach ($b in $branchesNeedingPr) {
    Write-Host "Comparing $b to $defaultBranch..."

    try {
        # GitHub compare API: base...head
        $compareJson = gh api "repos/$repo/compare/$defaultBranch...$b"
        $cmp = $compareJson | ConvertFrom-Json

        $results += [PSCustomObject]@{
            Branch = $b
            Ahead  = $cmp.ahead_by
            Behind = $cmp.behind_by
        }
    }
    catch {
        Write-Warning "Failed to compare ${b} to ${defaultBranch}: $($_.Exception.Message)"
    }
}

Write-Host ""
Write-Host "== Branch status vs $defaultBranch =="
if ($results.Count -gt 0) {
    $results |
        Sort-Object -Property @{ Expression = 'Ahead'; Descending = $true } |
        Format-Table Branch, Ahead, Behind -AutoSize
} else {
    Write-Host "(no non-default branches without open PRs)"
}

# 5) If -pr was passed, create PRs for branches that are ahead
if ($pr) {
    Write-Host ""
    Write-Host "== Creating pull requests for branches ahead of $defaultBranch =="

    # Only branches that are ahead of default
    $branchesToPr = $results | Where-Object { $_.Ahead -gt 0 }

    if (-not $branchesToPr) {
        Write-Host "No branches are ahead of $defaultBranch. Nothing to PR."
    } else {
        foreach ($item in $branchesToPr) {
            $branch = $item.Branch
            Write-Host "Creating PR for branch '$branch' (Ahead: $($item.Ahead), Behind: $($item.Behind))..."

            gh pr create `
                --base $defaultBranch `
                --head $branch `
                --title "$branch" `
                --body "Auto-created PR for branch '$branch' vs '$defaultBranch'."
        }
    }
}

# 6) ALWAYS show OPEN PRs and their mergeability
Write-Host ""
Write-Host "== Open pull requests on $repo (with mergeability) =="

# Re-fetch open PRs to include any just-created ones
$openPrsJson = gh pr list --state open --limit 100 --json number,title,headRefName,baseRefName
$openPrs = $openPrsJson | ConvertFrom-Json

$prStatus = @()

if ($openPrs) {
    foreach ($prItem in $openPrs) {
        $number = $prItem.number

        # Use REST API to get mergeable_state (clean/dirty/blocked/etc.)
        try {
            $prDetailJson = gh api "repos/$repo/pulls/$number"
            $prDetail = $prDetailJson | ConvertFrom-Json

            $mergeState = $prDetail.mergeable_state
            if (-not $mergeState) {
                $mergeState = "unknown"
            }

            # Human-friendly description
            switch ($mergeState) {
                "clean"   { $mergeDesc = "mergeable" }
                "dirty"   { $mergeDesc = "conflicts" }
                "blocked" { $mergeDesc = "blocked" }
                "unstable"{ $mergeDesc = "checks failing" }
                default   { $mergeDesc = $mergeState }
            }

            $prStatus += [PSCustomObject]@{
                Number       = $prItem.number
                HeadBranch   = $prItem.headRefName
                BaseBranch   = $prItem.baseRefName
                MergeState   = $mergeDesc   # pretty
                MergeRaw     = $mergeState  # raw value from API: clean/dirty/etc.
                Title        = $prItem.title
            }
        }
        catch {
            Write-Warning "Failed to get merge status for PR #${number}: $($_.Exception.Message)"
        }
    }

    if ($prStatus.Count -gt 0) {
        $prStatus |
            Sort-Object -Property Number |
            Format-Table Number, HeadBranch, BaseBranch, MergeState, Title -AutoSize
    } else {
        Write-Host "(no open pull requests found)"
    }
} else {
    Write-Host "(no open pull requests found)"
}

# 7) If -merge was passed, merge all clean/mergeable PRs
if ($merge -and $prStatus.Count -gt 0) {
    Write-Host ""
    Write-Host "== Merging open pull requests with clean/mergeable state =="

    # mergeable_state == "clean" → safe to merge
    $mergeablePrs = $prStatus | Where-Object { $_.MergeRaw -eq "clean" }

    if (-not $mergeablePrs) {
        Write-Host "No open PRs are currently clean/mergeable."
    } else {
        foreach ($m in $mergeablePrs) {
            $num = $m.Number
            Write-Host "Merging PR #$num ($($m.Title)) from '$($m.HeadBranch)' into '$($m.BaseBranch)'..."

            try {
                # Merge without deleting the branch
                gh pr merge $num --merge --body "Auto-merged by pr.ps1"
            }
            catch {
                Write-Warning "Failed to merge PR #${num}: $($_.Exception.Message)"
            }
        }
    }

    Write-Host ""
    Write-Host "== Pulling latest changes from $defaultBranch =="

    try {
        # Switch to the default branch
        git checkout $defaultBranch

        # Pull latest merged changes
        git pull origin $defaultBranch

        Write-Host "Successfully pulled latest $defaultBranch."
    }
    catch {
        Write-Warning "Failed to pull latest ${defaultBranch}: $($_.Exception.Message)"
    }
}

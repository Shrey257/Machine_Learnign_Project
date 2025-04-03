# PowerShell script to run football video analysis

# Set working directory to script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location -Path $scriptPath

# Define input/output paths
$inputVideo = "input/Raw_Data.mp4"
$outputVideo = "output/videos/final_result.mp4"
$modelPath = "models/football-players-detection/weights/best.pt"

# Run the analysis
Write-Host "Starting football video analysis..."
Write-Host "Input video: $inputVideo"
Write-Host "Output video: $outputVideo"
Write-Host "Model path: $modelPath"

# Run the command
python main.py --input $inputVideo --output $outputVideo --model $modelPath --report

Write-Host "Analysis complete. Check the output at $outputVideo"
Write-Host "Report generated at output/reports/player_stats_report.html"

# Open the output file when done
if (Test-Path $outputVideo) {
    Write-Host "Opening output video..."
    Invoke-Item $outputVideo
}

# Open the HTML report
$reportPath = "output/reports/player_stats_report.html"
if (Test-Path $reportPath) {
    Write-Host "Opening HTML report..."
    Invoke-Item $reportPath
} 
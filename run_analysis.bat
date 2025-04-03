@echo off
echo Starting football video analysis...

rem Navigate to script directory
cd /d %~dp0

rem Define paths
set INPUT_VIDEO=input\Raw_Data.mp4
set OUTPUT_VIDEO=output\videos\final_result.mp4
set MODEL_PATH=models\football-players-detection\weights\best.pt

echo Input video: %INPUT_VIDEO%
echo Output video: %OUTPUT_VIDEO%
echo Model path: %MODEL_PATH%

rem Run the analysis
python main.py --input %INPUT_VIDEO% --output %OUTPUT_VIDEO% --model %MODEL_PATH% --report

echo Analysis complete! Check the output at %OUTPUT_VIDEO%
echo Report generated at output\reports\player_stats_report.html

rem Open the output file if it exists
if exist %OUTPUT_VIDEO% (
    echo Opening output video...
    start %OUTPUT_VIDEO%
)

rem Open the HTML report if it exists
set REPORT_PATH=output\reports\player_stats_report.html
if exist %REPORT_PATH% (
    echo Opening HTML report...
    start %REPORT_PATH%
)

echo Done!
pause 
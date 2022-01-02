::::::::::::::::::::::::::::::::::::::::::::::::
:: DON'T EDIT ANYTHING BELOW
:: UNLESS YOU KNOW WHAT YOU'RE DOING
:: * LINES WITH :: IN FRONT ARE COMMENTS
::
:: // Command Prompt Syntax
:: >>> move-ocr.bat {MODE} "{SOURCE}" "{DESTINATION}"
::
:: // Example Usage
:: >>> move-oct.bat COPY "C:\Users\ZM\Desktop\ssmc-ocr\C drive" "C:\Users\ZM\Desktop\ssmc-ocr\M drive\2021"
::
:: // {MODE}
:: * Set mode to either MOVE/COPY
:: * Type in all capital letters
::
:: // {SOURCE/DESTINATION}
:: * Wrap each path in double quotes
:: * Make sure no \ symbol at the end
::
:: // Description
:: Moves or copies all image files from
::  specified source directory to destination.
:: Also creates folders (if not exist), 
::  based on the wafer IDs in the filename
::  eg. S9C998-..... (before the dash). 
::
::::::::::::::::::::::::::::::::::::::::::::::::

@echo off & setlocal EnableDelayedExpansion

set "MODE=%~1"
set "SOURCE=%~2"
set "DESTINATION=%~3"

if not %MODE% == MOVE if not %MODE% == COPY (
  echo MODE: %MODE%
  echo [ERROR] MODE variable was spelt incorrectly or not in capital letters
  pause
  exit /b
)

if not exist "%SOURCE%" (
  echo SOURCE: %SOURCE%
  echo [ERROR] SOURCE location not found, please check the paths again for spelling, wrapped in double quotes, no \ symbol at the end
  pause
  exit /b
)

if not exist "%DESTINATION%" (
  echo DESTINATION: %DESTINATION%
  echo [ERROR] DESTINATION location not found, please check the paths again for spelling, wrapped in double quotes, no \ symbol at the end
  pause
  exit /b
)

echo --- START ---

set /a "new_folders = 0"
set /a "affected_files = 0"

for /r "%SOURCE%" %%f in ("OCR Images\*.jpg") do (
  set "FULLPATH=%%f"

  set "FILENAME=%%~nf"
  echo FILENAME: !FILENAME!
  
  for /f "tokens=1 delims=-" %%a in ("!FILENAME!") do set ID=%%a
  echo ID: !ID!
  
  set "ID_FOLDER=%DESTINATION%\!ID!"
  if not exist "!ID_FOLDER!" (
    set /a "new_folders = !new_folders!+1"
    mkdir "!ID_FOLDER!"
  )
  
  if %MODE% == MOVE (
    move /y "!FULLPATH!" "!ID_FOLDER!"
  ) else if %MODE% == COPY (
    copy /y "!FULLPATH!" "!ID_FOLDER!"
  )
  
  set /a "affected_files = !affected_files!+1"
  
  echo;
)

echo --- END ---
echo // SUMMARY //
echo;
echo DATETIME: %date% %time%
echo MODE: %MODE%
echo SOURCE: %SOURCE%
echo DESTINATION: %DESTINATION%
echo NEW FOLDERS CREATED: %new_folders%
echo FILES AFFECTED: %affected_files%
echo;

pause
exit /b
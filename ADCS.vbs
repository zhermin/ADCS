Set WshShell = CreateObject("WScript.Shell" )
WshShell.Run chr(34) & ".\assets\run.bat" & Chr(34), 0
Set WshShell = Nothing
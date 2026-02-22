Set oShell = CreateObject("WScript.Shell")
oShell.Run "cmd /c ""cd /d C:\Users\neleftheriou\claude\congress_monitor && C:\Users\neleftheriou\AppData\Local\Programs\Python\Python312\python.exe monitor.py >> daemon.log 2>&1""", 0, False

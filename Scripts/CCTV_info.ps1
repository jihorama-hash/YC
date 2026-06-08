C:\ffmpeg\bin\ffprobe.exe -v error -select_streams v:0 `
-show_entries stream=width,height,sample_aspect_ratio,display_aspect_ratio `
-of default=noprint_wrappers=1 `
"record\20260528_CCTV001_yolo_5min01.mp4"
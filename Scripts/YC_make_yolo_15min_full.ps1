$ffmpeg = "C:\ffmpeg\bin\ffmpeg.exe"

$inputpath = "C:\Home\YC\record\20260528_CCTV001_yolo_5min01.mp4"
$output = "C:\Home\YC\record\for yolo\20260528_CCTV001_yolo_5min01.mp4"

& $ffmpeg -y `
  -ss 00:00:00 `
  -i $inputpath `
  -t 00:05:00 `
  -vf "fps=10,scale=1280:-2" `
  -an `
  -c:v libx264 `
  -preset veryfast `
  -crf 22 `
  -pix_fmt yuv420p `
  $output
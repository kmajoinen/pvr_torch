#!/usr/bin/env bash
# Extract N evenly spaced frames from a video window, centre-cropped to 4:3 at 480x360,
# for the dataset frame strip in the human-video chapter (fig:hv_dataset_frames).
#
# Usage: extract_frames.sh VIDEO OUT_PREFIX [N=4] [START=0] [END=end of video]
#   START and END are seconds (152.3) or timestamps (00:02:32.30, as in the EPIC-KITCHENS csv).
#
# Examples:
#   SSv2 clip, whole clip (pick the id and its label from something-something-v2-train.json):
#     ./extract_frames.sh <ssv2_root>/<id>.webm out/hv_ssv2
#   EPIC-KITCHENS-100 action segment (video_id, start_timestamp, stop_timestamp, narration from EPIC_100_train.csv):
#     ./extract_frames.sh <epic_root>/P01/videos/P01_01.MP4 out/hv_epic 4 00:02:32.30 00:02:35.90
#
# Writes OUT_PREFIX_1.jpg ... OUT_PREFIX_N.jpg. The chapter expects hv_epic_* and hv_ssv2_* in Overleaf's fig/.
set -euo pipefail

command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || {
  echo "ffmpeg/ffprobe not found: try 'module spider ffmpeg', or install ffmpeg from conda-forge into a conda env" >&2; exit 1; }

to_sec() {  # 152.3 -> 152.3 ; 00:02:32.30 -> 152.30
  awk -v t="$1" 'BEGIN{n=split(t,p,":"); s=0; for(i=1;i<=n;i++) s=s*60+p[i]; printf "%.3f", s}'
}

video=${1:?video path}
prefix=${2:?output prefix, e.g. out/hv_epic}
n=${3:-4}
start=$(to_sec "${4:-0}")
if [[ $# -ge 5 ]]; then
  end=$(to_sec "$5")
else
  end=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$video")
fi

[[ $n =~ ^[1-9][0-9]*$ ]] || { echo "N must be a positive integer, got '$n'" >&2; exit 1; }
awk -v s="$start" -v e="$end" 'BEGIN{exit !(e ~ /^[0-9.]+$/ && e + 0 > s + 0)}' ||
  { echo "need END > START, got START=$start END=$end (pass END explicitly if ffprobe has no duration)" >&2; exit 1; }

mkdir -p "$(dirname "$prefix")"
# frames at the centres of n equal sub-intervals, so the first and last frames are not black/transition frames
for i in $(seq 1 "$n"); do
  t=$(awk -v s="$start" -v e="$end" -v i="$i" -v n="$n" 'BEGIN{printf "%.3f", s + (e - s) * (i - 0.5) / n}')
  out="${prefix}_${i}.jpg"
  rm -f "$out"
  ffmpeg -nostdin -loglevel error -y -ss "$t" -i "$video" -frames:v 1 \
    -vf "scale=iw*sar:ih,setsar=1,crop='min(iw,ih*4/3)':'min(ih,iw*3/4)',scale=480:360" -q:v 2 "$out"
  [[ -s $out ]] || { echo "no frame at t = $t s in $video (past the end?)" >&2; exit 1; }
  echo "$out  (t = ${t} s)"
done

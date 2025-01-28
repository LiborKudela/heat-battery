#!/bin/bash

temp_dir=$(mktemp -d)
cache_file="assets/cloc/line_count_cache.csv"

if [[ ! -f $cache_file ]]; then
    echo "Cache file not found, creating new cache file"
    echo "date;commit;wc_total_chars;wc_total_lines;cloc_blank_lines;cloc_comments_lines;cloc_code_lines;cloc_total_lines;norm_pages;commit_msg;counted_files;skipped_files" > $cache_file

fi
cache_count=$(wc -l $cache_file | awk '{ print $1 }')
commits_to_process=$(git log --format="%H" --reverse | awk '{ print $1 }' | tail -n +$cache_count)
verbose=1;

rm -rf $temp_dir/files || echo "no files to remove.. OK"
mkdir -p $temp_dir/files
ignore_file=$temp_dir/ignore_file
found_file=$temp_dir/found_file

for commit in $commits_to_process; do
    date=$(git show -s --format=%ci $commit)
    msg=$(git show -s --format=%s $commit)
    if [[ $verbose -eq 1 ]]; then
        echo "procesing: $commit $date $msg"
    fi
    git archive "$commit" --output="$temp_dir/archive.tar"
    tar -x -C "$temp_dir/files" -f "$temp_dir/archive.tar"
    exclude_d_pattern="heat_battery/visualization/assets/(lotties|images)|assets/cloc"
    cloc_output=$(cloc "$temp_dir/files" --sum-one --quiet --fullpath --not-match-d="$exclude_d_pattern" --ignored="$ignore_file" --counted="$found_file" | grep SUM)
    cloc_blank_lines=$(echo "$cloc_output" | awk '{print $3}')
    cloc_comments_lines=$(echo "$cloc_output" | awk '{print $4}')
    cloc_code_lines=$(echo "$cloc_output" | awk '{print $5}')
    cloc_total_lines=$(($cloc_blank_lines + $cloc_comments_lines + $cloc_code_lines))
    readarray -t skipped_array < <(cat "$ignore_file" | awk -F':' '{print $1}')

    wc_output=$(xargs -a "$found_file" wc -lm | grep 'total')
    wc_total_lines=$(echo "$wc_output" | awk '{print $1}')
    wc_total_chars=$(echo "$wc_output" | awk '{print $2}')
    norm_pages=$(($wc_total_chars / 1800))
    rm -rf $temp_dir/files
    mkdir -p $temp_dir/files 

    counted_files=$(cat "$found_file" | xargs -I {} echo -n "{}, " | sed "s|$temp_dir/files/||g" | sed 's/, $//')
    skipped_files=$(cat "$ignore_file" | awk -F':' '{print $1}' | xargs -I {} echo -n "{}, " | sed "s|$temp_dir/files/||g" | sed 's/, $//')

    new_data=$(echo $date\;$commit\;$wc_total_chars\;$wc_total_lines\;$cloc_blank_lines\;$cloc_comments_lines\;$cloc_code_lines\;$cloc_total_lines\;$norm_pages\;$msg\;$counted_files\;$skipped_files)
    echo "$new_data" >> $cache_file

done
rm -rf $temp_dir
script_dir=$(dirname "${BASH_SOURCE[0]}")
python3 "$script_dir/create_cloc_chart.py"

#if [ -z "$3" ]
#  then
#    echo "No output dir specified."
#    exit
#fi

args_file=args/${1}.txt
output_dir=results/${3}
echo $args_file
mkdir -p ${output_dir}
condor_submit job.sub \
  args_file=${args_file} \
  num_jobs=${2:-1} \
  output_dir=${output_dir}

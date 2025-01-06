task=$1
#idx_start=$2
#idx_start=$3

for i in {20..30}
do
  python3 ./bin/test.py --filename /home/ito/08/eipl/eipl/tutorials/robosuite/sarnn/log/$1/SARNN.pth --idx "${i}" --task $1 --initial_position "${i}"
done
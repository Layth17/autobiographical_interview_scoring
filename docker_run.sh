# To run an interactive session, do "docker run -it ..."

docker run -v /home/layth/git/autobiographical_interview_scoring/test_data:/data \
  -e INPUT_DATA_DIR=/data/example_data.csv \
  -e OUTPUT_DIR=/data/output \
  aais > ./test_data/output/log.txt 2>&1

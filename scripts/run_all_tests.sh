echo "STARTING ALL TESTS:"
cd ..
for i in $(seq 1 $1);
do
  echo "Clearing cache"
  rm -rf cache
  SEED=$RANDOM
  echo "SEED:$SEED"
  scripts/output_tests.sh $SEED &
  BACK_PID=$!
  scripts/input_tests.sh $SEED &
  BACK_PID2=$!
  scripts/base_tests.sh $SEED &
  BACK_PID3=$!
  wait $BACK_PID && wait $BACK_PID2 && wait $BACK_PID3
done
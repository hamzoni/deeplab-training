0. Clean the source
./0_clean.sh

1. Run installation
./1_install.sh

2. Download label me to corp
python labelme/main.py

3. Run create folder structures
./2_structure.sh

4. Upload JSONs folder
./3_after_crop.sh card

5. Generate tfrecords
./4_tfrecord.sh

6. Training
./5_train.sh

7. Open tensorboard for monitoring
./6_board.sh

8. Export frozen graph

# Full Sequence 1 

First full sequence of radar images from [Oxford Robot Car Dataset](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/datasets/2019-01-10-11-46-21-radar-oxford-10k) (10/01/2019)

## Dataset Information

Time: 11:46:21 GMT
Duration: 00:37:00
Distance: 9.02 km

Frames: 8867
FPS: 4.00 Hz
Resolution: 4.32 cm
Range: 165 m

## Dataset Structure

```

../data/
 |-- full_seq_1/
      |-- radar/
         |-- (bunch of radar images labelled as <timestamp>.png)
      |-- info.txt (this file)
      |-- radar.timestamps (in form: <timestamp> <valid-bit>
      |-- gt
         |-- radar_odometry.csv (processed ground truth odometry data from GPS)

 ```

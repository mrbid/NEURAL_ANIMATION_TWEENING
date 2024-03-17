Slightly improved code, naming, and a much higher resolution mesh.

- Run the script in `mushman_exporter.blend` to create the `frames_ply` directory.
- Run the script in `scripts.blend` to create the `training_data` directory.
- Run `python3 fit.py` to train the network.
- Change directory to `models/tanh_adam_3_16_1_90_333_pd` and execute `CSVtoASC.sh` inside the directory to create `models/ASC`
- Drag one or multiple ASC files into [Meshlab](https://meshlab.net) to view the generated animation frame vertex data as a point cloud.

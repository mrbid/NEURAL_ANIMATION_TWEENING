**Slightly improved code, naming, and a much higher resolution mesh.**

- Run the script in `mushman_exporter.blend` to create the `frames_ply` directory.
- Run the script in `gen_training_data.blend` to create the `training_data` directory.
- Run `python3 fit.py` to train the network.
- Change directory to `models/selu_adam_6_32_6_90_777_pd` and execute `CSVtoASC.sh` inside the directory to create `models/ASC`
- Drag one or multiple ASC files into [Meshlab](https://meshlab.net) to view the generated animation frame vertex data as a point cloud.

**This system works better on smaller models or simpler animations, the mushman model has 25,000 vertices and has a complicated animation with alot of displacement which is too much chaos for a method like this to scale well or at all.**

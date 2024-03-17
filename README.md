This project takes a [vertex colored](https://github.com/VertexColor) and rigged animation in Blender, exports each animation frame as a PLY file, then converts the PLY files to CSV training data, then trains an MLP/FNN network with an input of 0-totalframes and output of a vertex buffer so that decimal inbetween frames can be requested from the network and it will generate the interpolated vertex data within a deviance of 0.002 of the original training data from a network that has 97,034 parameters (379.04 KB).

Basically a Feed-Forward Neural Network generates your 3D animation frames for you.

## Naming conventions
`girl_ply` - These are the exported frames for each step of the animation in [PLY format](https://paulbourke.net/dataformats/ply/).\
`girl_data` - This is the training data for the neural network.\
`models` - Data generated from the training process is saved here.\

## Steps
1. open `girl_rig_exporter.blend` and run the script `export_frames` and the `girl_ply` folder will be created of each animation frame.
2. open `scripts.blend` and run the script `ply_to_csv` and the `girl_data` folder will be created.
3. run `python3 fit.py` and the `girl_data` will be used to train a network which will be output to the `models` directory.
4. In the `models` directory will be a `*_pd` directory, cd into it and execute the `CSVtoASC.sh` file inside of the `*_pd` directory.
5. An `ASC` directory will now exist in the parent directory, inside here is a point cloud file in the .asc format of each vertex
for each frame of the animation, you can load these into meshlab.

The `*_pd` directory contains test prediction data from the trained network for every frame and three frames between 0, 0.25, 0.50, 0.75..

Ultimately you will want to export the trained network weights and use them in your program to generate the output vertices in real-time
based on a variable floating point input that represents the current time point between two animation frames.

## Reality Check
Why would anyone want to do this?

There are 100 frames of training data, but in reality that is only 10 frames that would be linearly interpolated between in a vertex
shader. Each frame is ~22.63 KB in vertex data, so 10 frames is only 226.32 KB. This trained network provided as-is is 379.04 KB.

Furthermore the amount of multiplications and additions used in this network is far far higher by an order of magnitude than a simple
linear interpolate between frames and it's producing a much less accurate result.

Finally the network weights probably compress less well than the traditional 10 frames would even if they where the same starting size.

There is no benefit to using a neural network to generate vertex data for your mesh animations over lerping between animation frames,
or even better, using a quaternion based skeletal animation system.

But, it's pretty cool that it works, it's not a huge incrase in disk/ram space and the quality loss is not visually that bad.

It's a loss I'd be willing to take just to be different.

Although I would run the network on a CPU with FMA auto-vectorisation rather than in a shader... Which again could be seen as another loss
as you'd have to send each frame from the CPU over to the GPU each time; where as the traditional method easily all happens in a vertex shader
on data already loaded into GPU memory.


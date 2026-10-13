# Required assets

Only the image-conditioned examples require input assets. All files are stored locally in this directory.

| Task/backbone | Image | 3D mesh |
|---|---|---|
| Wan2.2-TI2V-5B (I2V) | [wan22_5b.jpeg](i2v/wan22_5b.jpeg) | — |
| Wan2.2-I2V-A14B | [wan22_14b.jpeg](i2v/wan22_14b.jpeg) | — |
| Cosmos-Predict2.5-2B | [cosmos_predict2_5.jpeg](i2v/cosmos_predict2_5.jpeg) | — |
| Hunyuan3D-2, sample 143 | [143.jpeg](i2_3d/143.jpeg) | [143.glb](i2_3d/143.glb) |
| Hunyuan3D-2, sample 209 | [209.jpeg](i2_3d/209.jpeg) | [209.glb](i2_3d/209.glb) |
| Hunyuan3D-2, sample 386 | [386.jpeg](i2_3d/386.jpeg) | [386.glb](i2_3d/386.glb) |

The scripts resolve these paths relative to the repository, so the defaults work independently of the current working directory. Pass `--image` or `--mesh` to use other local assets.

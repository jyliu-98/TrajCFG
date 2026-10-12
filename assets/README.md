# Qualitative assets

Each example is selected from the paper's non-VBench qualitative results. The media is hosted in the project-page repository to avoid storing duplicate binary files in the code repository.

| Task/backbone | Case | Input | TrajCFG result |
|---|---|---|---|
| Wan2.1-T2V-1.3B | Car splashing through a puddle (Dynamic-Bench) | Text | [Video](https://jyliu-98.github.io/trajcfg-project-page/assets/gallery/t2v/dynamic_bench_wan21_13b/trajcfg.mp4) |
| Wan2.2-TI2V-5B (T2V) | Cat interacting with a yarn ball (Dynamic-Bench) | Text | [Video](https://jyliu-98.github.io/trajcfg-project-page/assets/gallery/t2v/dynamic_bench_wan22_5b/trajcfg.mp4) |
| Wan2.2-T2V-A14B | Knife slicing a tomato (Dynamic-Bench) | Text | [Video](https://jyliu-98.github.io/trajcfg-project-page/assets/gallery/t2v/dynamic_bench_wan22_14b/trajcfg.mp4) |
| Wan2.2-TI2V-5B (I2V) | Robot-to-human cup handoff (PAI-Bench-G) | [Image](https://jyliu-98.github.io/trajcfg-project-page/assets/gallery/i2v/pai_g_wan22_5b/input.jpeg) | [Video](https://jyliu-98.github.io/trajcfg-project-page/assets/gallery/i2v/pai_g_wan22_5b/trajcfg.mp4) |
| Wan2.2-I2V-A14B | Robotic arm picking up a bowl (PAI-Bench-G) | [Image](https://jyliu-98.github.io/trajcfg-project-page/assets/gallery/i2v/pai_g_wan22_14b/input.jpeg) | [Video](https://jyliu-98.github.io/trajcfg-project-page/assets/gallery/i2v/pai_g_wan22_14b/trajcfg.mp4) |
| Cosmos-Predict2.5-2B | Honey dripping onto a honeycomb (PAI-Bench-G) | [Image](https://jyliu-98.github.io/trajcfg-project-page/assets/gallery/i2v/pai_g_cosmos/input.jpeg) | [Video](https://jyliu-98.github.io/trajcfg-project-page/assets/gallery/i2v/pai_g_cosmos/trajcfg.mp4) |

The I2V scripts use the raw GitHub URLs for these conditioning images by default. Pass `--image /path/to/input.jpg` to run entirely from local files.

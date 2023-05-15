# neural-networks-playground

A playground for Simulations in Natural Sciences project revolving around Artificial Neural Networks

## GPU Acceleration

To make learning faster, I am using my AMD gpu and it's HIP capabilites, [link](https://docs.amd.com/bundle/ROCm-Deep-Learning-Guide-v5.4.3/page/Frameworks_Installation.html#d2839e1290).

For that I will be using their docker setup since fedora is a pain to install ROCm for.

```bash
docker pull rocm/pytorch:latest
```

> **WARNING**: This will use **35GB of your storage!!!**

### Alternative

I found that using the conda environment included in the folders and then installing pytorch with `rocm` support works easiest.

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2`

## Resources

- [3blue1brown Neural Networks Intro](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [YOLOv3 from scratch](https://www.youtube.com/watch?v=Grir6TZbc1M)
  - [Aladding Persson's ML collection](https://github.com/aladdinpersson/Machine-Learning-Collection)

## TODO

- goal is to do real-time object detection
  - use the persson tutorial for that

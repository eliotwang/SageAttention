docker run -it --network=host -v /home/hnl:/home  --device=/dev/kfd --device=/dev/dri --group-add video  --security-opt seccomp=unconfined --ipc=host --name sgattn lmsysorg/sglang:v0.4.7-rocm630 

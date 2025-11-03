## Summary

This is the code for the paper **EdinoGait: Transferring Large Visual Models to Event-based Vision for Enhancing Gait Recognition** by Liaogehao Chen , Zhenjun Zhang , and Yaonan Wang.

If you use any of this code, please cite the following publication:

> ```
> @article{chen2025edinogait,
>     title={EdinoGait: Transferring Large Visual Models to Event-based Vision for Enhancing Gait Recognition},
>     author={Chen, Liaogehao and Zhang, Zhenjun and Wang, Yaonan},
>     journal={IEEE Transactions on Multimedia},
>     year={2025},
>     publisher={IEEE}
> }
> ```

# Requirements

For details, please see https://github.com/ShiqiYu/OpenGait/blob/master/docs/0.get_started.md

# Data

- **DAVIS346-Gait**: https://pan.baidu.com/s/1joc62krCik4rsoItncdlRA?pwd=yy26 , extraction code: **yy26**

# Preprocess

- Download the [pre-trained model](https://drive.google.com/drive/folders/1blvrnbZP3IPNOOYhQzdz5Ml4z-vnKBb9?usp=drive_link), and place it in the `./pretrained` folder
- Download the **DAVIS346-Gait** dataset and place the dataset into the `./db` directory. Run `python ./datasets/bboxes.py` to obtain bounding boxes. Then run `python ./datasets/preprocess.py` and `python ./datasets/to_cef.py` sequentially.



# Train

Train Edino by

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 opengait/main.py --cfgs ./configs/edinov2_cef_hnu.yaml
```

Place the trained Edino into the `./pretrained/` directory, or use the pre-trained Edino provided by us. Train EdinoGait  by

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/edinov2_reg_cef_hnugait_light.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 opengait/main.py --cfgs ./configs/edinov2_reg_cef_hnugait_dark.yaml
```

## Acknowledgement

[OpenGait](https://github.com/ShiqiYu/OpenGait)

[BigGait](https://github.com/ShiqiYu/OpenGait)

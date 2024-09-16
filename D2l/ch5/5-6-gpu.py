import torch
from d2l import torch as d2l
nums = 100
x_cpu = torch.ones(nums,nums)
x_gpu = torch.ones(nums,nums, device='cuda')

timer1 = d2l.Timer()
torch.mm(x_cpu,x_cpu)
print("cpu cost time {:.5f}".format(float(timer1.stop())))

timer2 = d2l.Timer()
torch.mm(x_gpu,x_gpu)
print("gpu cost time {:.5f}".format(float(timer2.stop())))


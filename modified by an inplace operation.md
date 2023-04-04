# "one of the variables needed for gradient computation has been modified by an inplace operation".  

This is a bug that cost my whole afternoon to solve.  

# Description：  
RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [4, 512, 16, 16]], which is output 0 of ConstantPadNdBackward, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).  

# Bug Location:
loss.backward()  

# Solution:  
As I searched on the internet, there are several ways to fix with recommandation:  

1. If there are 'inplace=True' in code, change them to 'inplace=False'. Modules like ReLU have this parameter.
2. If there are operator like '+=', '-=' and so on, change them to 'A = A + B'. These are also inplace operations.
3. If there 'nn.parallel.DistributedDataParallel' in code, change the 'broadcast_buffers' parameter as 'broadcast_buffers=False'. In distributed training, if the same model is called multiple times, above error will be triggered. The 'forword()' and 'backward()' functions must be executed interalterably and will throw an error if you execute 'forward()' multiple times and then 'backward()' once. 
4. The ' optimizer.step() ' function of the training code should be put behind ' loss.backward() '.
5. My situation occured in following code: 'Nodes_list[batch] = GNN(A_list[batch], Nodes_list[batch], nodes_mask_list)'. This is usually the case when the tensor is being manipulated without clone(). For example, if your tensor with gradient information takes the 'sin' transform, you would write 'img = torch.sin(img.clone())'. If you just wrote 'img = torch.sin(img)', you'd get an error. Here the variable 'Nodes_list' was overwrote for several times. I changed batch_size and I found the *version X, expected version X-1* was mentioned and X will change with batchsize. So I changed it to 'Nodes_list[batch] = GNN(A_list[batch], Nodes_list[batch].clone, nodes_mask_list)'. It works.

import torch


#设置一个随机种子
torch.manual_seed(100) 
#生成一个形状为2x3的矩阵
x = torch.randn(2, 3)
print(x)
#根据索引获取第1行，所有数据
print(x[0,:])
#获取最后一列数据
print(x[:,-1])
#生成是否大于0的Byter张量
mask=x>0
print(mask)
#获取大于0的值
print(torch.masked_select(x,mask))
#获取非0下标,即行，列索引
print(torch.nonzero(mask))
#获取指定索引对应的值,输出根据以下规则得到
#out[i][j] = input[index[i][j]][j]  # if dim == 0
#out[i][j] = input[i][index[i][j]]  # if dim == 1
index=torch.LongTensor([[0,1,1]])
print(index)
print(torch.gather(x,0,index))  #https://blog.csdn.net/cpluss/article/details/90260550
index=torch.LongTensor([[0,1,1],[1,1,1]])
a=torch.gather(x,1,index)
print(a)
#把a的值返回到一个2x3的0矩阵中
z=torch.zeros(2,3)
print(z)
print(z.scatter_(1,index,a))
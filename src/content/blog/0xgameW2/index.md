---
title: '0xGame 2024 Week2 Writeup'
publishDate: 2024-10-18
description: '0xGame 2024 Week2 Writeup'
tags:
  - CTF
language: '中文'
# heroImage: { src: './thumbnail.jpg', color: '#D58388' }
---

# BabyUPX

使用UPX脱壳工具之后打开，从encode函数发现加密逻辑是交换字节的高四位和第四位，可以写出exp

```python
def decode(encoded_bytes):
    decoded_bytes = encoded_bytes
    
    for i in range(len(decoded_bytes)):
        high = decoded_bytes[i] & 0xF0
        low = decoded_bytes[i] & 0x0F
        decoded_bytes[i] = (low << 4) | (high >> 4)
    return decoded_bytes
encoded_data = [0x03, 0x87, 0x74, 0x16, 0xD6, 0x56, 0xB7, 0x63, 0x83, 0x46, 0x66, 0x66, 0x43, 0x53, 0x83, 0xD2, 0x23, 0x93, 0x56, 0x53, 0xD2, 0x43, 0x36, 0x36, 0x03, 0xD2, 0x16, 0x93, 0x36, 0x26, 0xD2, 0x93, 0x73, 0x13, 0x66, 0x56, 0x36, 0x33, 0x33, 0x83, 0x56, 0x23, 0x66, 0xD7]
decoded_data = decode(encoded_data)
flag = ''
for i in range(len(decoded_data)):
    flag += chr(decoded_data[i])
print(flag)
#0xGame{68dff458-29e5-4cc0-a9cb-971fec338e2f}
```

# FirstSight-Jar

使用Jadx打开jar文件，在main函数中能看到uuid，拼接上0xGame即可得到flag

# FirstSight-Pyc

使用pydumpck反编译pyc文件得到py文件逻辑

```python

import hashlib
user_input = input('请输入神秘代号：')
if user_input != 'Ciallo~':
    print('代号不是这个哦')
    exit()
    input_hash = hashlib.md5(user_input.encode()).hexdigest()
    input_hash = list(input_hash)
for i in range(len(input_hash)):
    if ord(input_hash[i]) in range(48, 58):
        original_num = int(input_hash[i])
        new_num = (original_num + 5) % 10
        input_hash[i] = str(new_num)
    else:
        input_hash = ''.join(input_hash)
        print('0xGame{{{}}}'.format(input_hash))
		return None

```

修改为正确的逻辑输入Ciallo~即可得到flag

0xGame{2f0ef0217bf3a7c598d381b077672e09}

# ZzZ

使用IDA打开后看到flag具体格式以及中间部分的加密逻辑，使用Z3求解方程得到flag

```python
from z3 import *

def solve_for_flag():

    v10 = BitVec('v10', 64)
    v11 = BitVec('v11', 64)
    v12 = BitVec('v12', 64)

    solver = Solver()

    solver.add(11 * v11 + 14 * v10 - v12 == 0x48FB41DDD)
    solver.add(9 * v10 - 3 * v11 + 4 * v12 == 0x2BA692AD7)
    solver.add(((v12 - v11) >> 1) + (v10 ^ 0x87654321) == 3451779756)


    if solver.check() == sat:
        model = solver.model()
        v10_value = model[v10].as_long()
        v11_value = model[v11].as_long()
        v12_value = model[v12].as_long()

        print(f"v10: {v10_value}, v11: {v11_value}, v12: {v12_value}")

    v5 = v10_value.to_bytes(4, byteorder='little').decode('utf-8')
    v6 = v11_value.to_bytes(4, byteorder='little').decode('utf-8')
    v7 = v12_value.to_bytes(4, byteorder='little').decode('utf-8')


    v13 = 0xe544267d      
    v14 = 0xd085a85201a4    

    flag = "0xGame{{{:08x}-{}-{}-{}-{:012x}}}".format(v13, v5, v6, v7, v14)
    
    print(flag)
# 0xGame{e544267d-2187-3b44-d53a-d085a85201a4}
solve_for_flag()

```

# Xor::Ramdom

有关伪随机数的异或

理清代码逻辑后可以在init_random函数中发现随机数的种子

```c
int init_random(void)
{
  srand(0x77u);
  return rand();
}
```

同时我们可以在main函数中看到v21 = rand( )，所以一共调用了两次rand函数，接下来是一个简单的逻辑，根据索引的奇偶性来判断实际的异或值

```C
if ( (v23 & 1) != 0 )
      v8 = v21;
    else
      v8 = v21 + 3;
    *v7 ^= v8;
```

根据以上分析可以写出exp

```c++
#include<stdio.h>
#include<stdlib.h>

int main()
{
	srand(0x77u);
	int cipher[]{0x0c,0x4f,0x10,0x1f,0x4e,0x16,0x21,0x12,0x4b,0x24,0x10,0x4b,0x0a,0x24,0x1f,0x17,0x09,0x4f,0x07,0x08,0x21,0x5c,0x2c,0x1a,0x10,0x1f,0x11,0x16,0x59,0x5a};
 	int random_number1 = rand()%256;
 	int random_number2 = rand()%256;
	printf("0xGame{");
 	for(int i = 0; i < sizeof(cipher)/sizeof(cipher[0]);i++)
 	{
		if ((i&1)!=0)
			cipher[i] ^= random_number2;
		else
			cipher[i] ^= (random_number2+3);
		printf("%c",cipher[i]);
 	}
 	printf("}");
 	return 0;
 	//0xGame{r4nd0m_i5_n0t_alw4ys_'Random'!}
 } 
 
```

